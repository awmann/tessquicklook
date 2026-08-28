"""TESS Pixel Response Function: retrieval, resampling and fitting.

Ports ``gettessprf.pro``, ``resampletessprf.pro`` and ``mpfittessprf.pro``.

The official PRF library is fetched lazily from MAST and cached on disk, which
replaces the IDL's dependency on a pre-staged local PRF tree plus its
``prfinfo.idl`` index -- the grid is recovered from the filenames instead.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.optimize import least_squares

__all__ = [
    "prf_cache_dir",
    "get_tess_prf",
    "resample_tess_prf",
    "fit_tess_prf",
    "cubic_convolution_interpolate",
]

_MAST_PRF = "https://archive.stsci.edu/missions/tess/models/prf_fitsfiles"
# Filenames look like tess2019107181900-prf-1-1-row0001-col0045.fits
_FNAME_RE = re.compile(r"-prf-(\d)-(\d)-row(\d+)-col(\d+)\.fits$")

# The two PRF epochs.  The IDL maps sectors 1-3 to the first and 4+ to the
# second; MAST's own README says the updated models are valid from sector 4.
_EPOCHS = {1: "start_s0001", 4: "start_s0004"}


def prf_cache_dir() -> Path:
    d = Path(os.environ.get("TESSQUICKLOOK_CACHE", Path.home() / ".tessquicklook"))
    return d / "prf"


def _epoch_for_sector(sector: int) -> str:
    return _EPOCHS[1] if sector is not None and sector <= 3 else _EPOCHS[4]


def _list_remote_grid(epoch: str, camera: int, ccd: int):
    """Return ``[(row, col, filename), ...]`` for one CCD's PRF grid.

    The listing is static, so it is cached on disk as JSON.  Before that cache
    existed a single transient timeout here would abort a whole sector -- and
    because the batch driver suppresses warnings, do so silently.  Retried a few
    times, then cached permanently.
    """
    import json
    import urllib.error
    import urllib.request

    cache = prf_cache_dir() / epoch / f"cam{camera}_ccd{ccd}" / "_grid.json"
    if cache.exists():
        try:
            return [tuple(x) for x in json.loads(cache.read_text())]
        except Exception:  # noqa: BLE001
            pass  # corrupt cache: refetch

    url = f"{_MAST_PRF}/{epoch}/cam{camera}_ccd{ccd}/"
    html = None
    last_exc = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                html = resp.read().decode("utf-8", "replace")
            break
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_exc = exc
            if attempt < 3:
                warnings.warn(
                    f"PRF grid listing for cam{camera}/ccd{ccd} failed "
                    f"({type(exc).__name__}: {exc}); retry {attempt + 1}/3"
                )
    if html is None:
        raise OSError(
            f"Could not list PRF grid for {epoch} cam{camera} ccd{ccd}: {last_exc}"
        )

    out = []
    for match in re.finditer(r'href="([^"]+\.fits)"', html):
        name = match.group(1)
        m = _FNAME_RE.search(name)
        if m and int(m.group(1)) == camera and int(m.group(2)) == ccd:
            out.append((int(m.group(3)), int(m.group(4)), name))
    out = sorted(set(out))

    if out:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(out))
        except Exception:  # noqa: BLE001
            pass  # caching is an optimisation, never a hard requirement
    return out


def _ensure_prf_file(epoch: str, camera: int, ccd: int, filename: str) -> Path:
    dest = prf_cache_dir() / epoch / f"cam{camera}_ccd{ccd}" / filename
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    # Shares the quaternion downloader: bounded reads, .part staging, resume.
    from .systematics import _fetch_large_file

    url = f"{_MAST_PRF}/{epoch}/cam{camera}_ccd{ccd}/{filename}"
    return _fetch_large_file(url, dest)


_GRID_CACHE: dict = {}


def get_tess_prf(camera, ccd, rowcenter, colcenter, sector):
    """Port of ``gettessprf.pro`` (the non-``/blend`` branch).

    Selects the PRF grid point nearest the target's CCD position -- exactly
    what the IDL does via ``min(sqrt((info.row-row)^2 + (info.column-col)^2))``.

    Returns ``(prf, x, y, nsamp)``: the oversampled PRF image (indexed
    ``[row, col]``) and its pixel-offset axes.

    Note on ``/blend``: the IDL's bilinear branch reads ``filename11`` into all
    four corners, so its interpolation collapses to a single grid point anyway.
    Callers wanting blending get the same nearest-neighbour behaviour here.
    """
    epoch = _epoch_for_sector(sector)
    key = (epoch, int(camera), int(ccd))
    if key not in _GRID_CACHE:
        _GRID_CACHE[key] = _list_remote_grid(epoch, int(camera), int(ccd))
    grid = _GRID_CACHE[key]
    if not grid:
        raise RuntimeError(f"No PRF files found for {epoch} cam{camera} ccd{ccd}")

    rows = np.array([g[0] for g in grid], dtype=float)
    cols = np.array([g[1] for g in grid], dtype=float)
    dist = np.hypot(rows - float(rowcenter), cols - float(colcenter))
    best = int(np.argmin(dist))

    path = _ensure_prf_file(epoch, int(camera), int(ccd), grid[best][2])
    with fits.open(path) as hdul:
        prf = np.asarray(hdul[0].data, dtype=float)
        hdr = hdul[0].header

    nsamp = float(hdr.get("NSAMP", 9.0))
    x = hdr["CDELT1P"] * (np.arange(prf.shape[1]) - hdr["CRPIX1P"])
    y = hdr["CDELT2P"] * (np.arange(prf.shape[0]) - hdr["CRPIX2P"])
    return prf, x, y, nsamp


def cubic_convolution_interpolate(image, xi, yi, a=-0.5):
    """IDL ``INTERPOLATE(..., CUBIC=-0.5)``: Keys cubic *convolution*.

    scipy's ``map_coordinates(order=3)`` is a cubic *B-spline* interpolation,
    which is a genuinely different operator, so the Keys kernel is implemented
    directly here to keep the PRF model faithful.

    ``image`` is indexed ``[row, col]``; ``xi`` indexes columns and ``yi`` rows,
    matching IDL's ``[col, row]`` array convention.  Coordinates are clamped to
    the array edge, as IDL does when ``MISSING`` is not supplied.
    """
    img = np.asarray(image, dtype=float)
    ny, nx = img.shape
    xi = np.asarray(xi, dtype=float)
    yi = np.asarray(yi, dtype=float)

    x0 = np.floor(xi).astype(int)
    y0 = np.floor(yi).astype(int)
    fx = xi - x0
    fy = yi - y0

    def weights(f):
        # Keys kernel evaluated at the four taps offset -1, 0, +1, +2.
        f = f[..., None]
        s = np.abs(np.stack([f + 1.0, f, f - 1.0, f - 2.0], axis=-1)[..., 0, :])
        w = np.where(
            s <= 1.0,
            (a + 2.0) * s**3 - (a + 3.0) * s**2 + 1.0,
            a * s**3 - 5.0 * a * s**2 + 8.0 * a * s - 4.0 * a,
        )
        return np.where(s < 2.0, w, 0.0)

    wx = weights(fx)
    wy = weights(fy)

    out = np.zeros(np.broadcast(xi, yi).shape, dtype=float)
    for j in range(4):
        yy = np.clip(y0 + j - 1, 0, ny - 1)
        acc = np.zeros_like(out)
        for i in range(4):
            xx = np.clip(x0 + i - 1, 0, nx - 1)
            acc = acc + wx[..., i] * img[yy, xx]
        out = out + wy[..., j] * acc
    return out


def resample_tess_prf(x, y, psf, deltax, deltay, nsamp=9.0):
    """Port of ``resampletessprf.pro``.

    Evaluates the oversampled PRF on the detector pixel grid ``(x, y)`` for a
    star displaced by ``(deltax, deltay)`` from the centre of that grid.

    The IDL's ``offsetx = ncols(psf)/nsamp/2`` puts the origin at 58.5 samples
    for a 117-sample PRF whose true centre index is 58 -- a constant half-sample
    offset.  It is reproduced here rather than corrected, because the fitted
    centroid absorbs it and the aperture masks are built from the same model.
    """
    psf = np.asarray(psf, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    nosy, nosx = psf.shape
    offsetx = nosx / float(nsamp) / 2.0
    offsety = nosy / float(nsamp) / 2.0

    xi = (x - x.mean() + offsetx - deltax) * float(nsamp)
    yi = (y - y.mean() + offsety - deltay) * float(nsamp)
    return cubic_convolution_interpolate(psf, xi, yi)


def fit_tess_prf(xes, yes, flux, psf, p0, fit_center=True, nsamp=9.0):
    """Port of the ``mpfit('mpfittessprf', ...)`` call in extractphotometrytess.

    Model: ``amp * PRF(x - xc, y - yc) + dc``.  IDL weights residuals by
    ``e = sqrt(1 + |flux|)`` and holds the centre fixed unless a search radius
    was supplied (``srin``), both reproduced here.

    ``p0`` is ``[amp, xc, yc, dc]``.  Returns the fitted parameter vector.
    """
    xes = np.asarray(xes, dtype=float).ravel()
    yes = np.asarray(yes, dtype=float).ravel()
    flux = np.asarray(flux, dtype=float).ravel()
    err = np.sqrt(1.0 + np.abs(flux))

    p0 = np.asarray(p0, dtype=float)

    if fit_center:
        free = np.array([0, 1, 2, 3])
    else:
        free = np.array([0, 3])

    def unpack(theta):
        p = p0.copy()
        p[free] = theta
        return p

    def resid(theta):
        amp, xc, yc, dc = unpack(theta)
        model = amp * resample_tess_prf(xes, yes, psf, xc, yc, nsamp=nsamp) + dc
        return (flux - model) / err

    # amp and dc are constrained non-negative, as parinfo does in the IDL.
    lo = np.full(free.size, -np.inf)
    hi = np.full(free.size, np.inf)
    for k, idx in enumerate(free):
        if idx in (0, 3):
            lo[k] = 0.0

    res = least_squares(resid, np.clip(p0[free], lo, hi), bounds=(lo, hi), method="trf")
    return unpack(res.x)
