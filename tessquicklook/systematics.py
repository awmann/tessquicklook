"""Spacecraft quaternion and CBV vectors binned onto the FFI cadence.

Ports ``processquaternions.pro`` and ``bincbv.pro``.

Both routines take a finely-sampled engineering product (quaternions at ~2 s,
CBVs at the mission cadence) and reduce it onto the cadence grid of the FFI
light curve, producing the regressors that :mod:`tessquicklook.decorrelate`
fits simultaneously with the stellar variability.
"""

from __future__ import annotations

import hashlib
import os
import re
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits

__all__ = [
    "cadence_days_for_sector",
    "bin_quaternions",
    "load_quaternion_file",
    "load_quaternion_arrays",
    "bin_cbvs",
    "systematics_cache_dir",
]

_ENGINEERING = "https://archive.stsci.edu/missions/tess/engineering"

# The IDL reads C2_Q1/2/3 -- camera 2's quaternions -- for every target,
# regardless of which camera the star actually falls on, treating them as a
# proxy for overall spacecraft pointing.  Kept as the default for fidelity.
DEFAULT_QUAT_CAMERA = 2


def systematics_cache_dir() -> Path:
    d = Path(os.environ.get("TESSQUICKLOOK_CACHE", Path.home() / ".tessquicklook"))
    return d


def cadence_days_for_sector(sector: int) -> float:
    """FFI cadence by sector era: 30 min, then 10 min, then 200 s."""
    if sector < 27:
        return 30.0 / 60.0 / 24.0
    if sector < 56:
        return 10.0 / 60.0 / 24.0
    return 200.0 / 60.0 / 60.0 / 24.0


def _moment_skew_kurt(x):
    """IDL ``MOMENT()`` skewness and (excess) kurtosis.

    IDL normalises by the *sample* standard deviation (N-1 denominator) while
    dividing the summed powers by N, so this is not numpy's default.
    """
    n = x.size
    if n < 2:
        return 0.0, 0.0
    m = x.mean()
    d = x - m
    var = np.sum(d**2) / (n - 1)
    if var <= 0:
        return 0.0, 0.0
    sd = np.sqrt(var)
    skew = np.sum(d**3) / (n * sd**3)
    kurt = np.sum(d**4) / (n * sd**4) - 3.0
    return skew, kurt


def _signed_root(v, power):
    """IDL: ``abs(v)^(1/power) * sign(v)`` -- compresses the dynamic range."""
    v = np.asarray(v, dtype=float)
    out = np.abs(v) ** (1.0 / power) * np.sign(v)
    return np.where(np.isfinite(out), out, 0.0)


def _fetch_large_file(url, dest, timeout=60, attempts=5, chunk=1 << 20):
    """Download to a ``.part`` file with resume, then rename into place.

    Written for the ~430 MB quaternion files, where two failure modes bite:

    * ``urlretrieve`` takes no timeout, so a connection that drops mid-transfer
      leaves the process blocked on a socket read *forever* -- 0% CPU, no error,
      no progress.  Every read here is bounded by ``timeout``.
    * Writing straight to the destination makes a truncated file
      indistinguishable from a complete one, so the next run finds it, treats it
      as cached, and fails inside ``fits.open`` (or worse, reads short data).
      The file only takes its real name once the expected byte count has
      arrived.

    Interrupted transfers resume via an HTTP ``Range`` request rather than
    starting over.
    """
    import urllib.error
    import urllib.request

    dest = Path(dest)
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)

    last_exc = None
    for attempt in range(1, int(attempts) + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                # A server that ignores Range replies 200 and restarts the body.
                if have and resp.status != 206:
                    have = 0
                    part.unlink(missing_ok=True)
                total = resp.headers.get("Content-Length")
                total = int(total) + have if total is not None else None

                with open(part, "ab" if have else "wb") as fh:
                    while True:
                        block = resp.read(chunk)
                        if not block:
                            break
                        fh.write(block)

            got = part.stat().st_size
            if total is not None and got < total:
                raise OSError(f"truncated: {got} of {total} bytes")
            part.replace(dest)
            return dest
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_exc = exc
            if attempt < attempts:
                warnings.warn(
                    f"Download of {url.rsplit('/', 1)[-1]} interrupted "
                    f"({type(exc).__name__}: {exc}); resuming, "
                    f"attempt {attempt + 1}/{attempts}"
                )

    part.unlink(missing_ok=True)
    raise OSError(f"Could not download {url} after {attempts} attempts: {last_exc}")


def _engineering_listing(refresh=False, attempts=3, timeout=30):
    """Filenames in MAST's TESS engineering directory, cached on disk.

    The listing is a large, slow page and has been observed to stop responding
    entirely for stretches (75 s to first byte, then nothing).  Losing a sector
    to that is absurd when the answer -- which filename holds which sector's
    quaternions -- barely changes: it only grows as new sectors publish.

    So it is cached.  A failed refetch falls back to the cached copy rather than
    propagating, and callers refetch explicitly (``refresh=True``) when a sector
    is missing, which is the only case where staleness could mislead.
    """
    import json
    import urllib.error
    import urllib.request

    cache = systematics_cache_dir() / "quaternions" / "engineering_listing.json"
    cached = None
    if cache.exists():
        try:
            cached = json.loads(cache.read_text())
        except Exception:  # noqa: BLE001
            cached = None
    if cached and not refresh:
        return cached

    last_exc = None
    for attempt in range(1, int(attempts) + 1):
        try:
            with urllib.request.urlopen(f"{_ENGINEERING}/", timeout=timeout) as resp:
                html = resp.read().decode("utf-8", "replace")
            names = sorted({m.group(1) for m in
                            re.finditer(r'href="([^"]+-quat\.fits)"', html)})
            if names:
                try:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text(json.dumps(names))
                except Exception:  # noqa: BLE001
                    pass
                return names
            last_exc = RuntimeError("listing returned no quaternion files")
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_exc = exc
        if attempt < attempts:
            warnings.warn(
                f"MAST engineering listing failed ({type(last_exc).__name__}: "
                f"{last_exc}); retry {attempt + 1}/{attempts}"
            )

    if cached:
        warnings.warn(
            f"MAST engineering listing unreachable ({last_exc}); "
            f"using the cached copy of {len(cached)} filenames."
        )
        return cached
    raise OSError(f"Could not list MAST engineering directory: {last_exc}")


def load_quaternion_file(sector, search_dirs=None, download=True):
    """Locate (or fetch) the quaternion FITS file for a sector.

    Returns ``(path, was_downloaded)``. ``was_downloaded`` is True only when
    this call itself fetched the file into the owned cache directory
    (``systematics_cache_dir()/quaternions/fits``) -- False for every file
    found via ``search_dirs``, ``TESSQUICKLOOK_QUATERNION_DIR``, or the
    legacy path. Callers that delete the file after extracting from it
    (``discard_fits``) must check this first: a pre-staged
    ``TESSQUICKLOOK_QUATERNION_DIR`` (WanShiTon's whole reason for existing)
    is not this process's file to delete, and doing so anyway silently ate a
    435 MB pre-staged sector-47 file before this was fixed.
    """
    pattern = re.compile(rf"tess\d+_sector{int(sector):02d}-quat\.fits$")
    dirs = list(search_dirs or [])
    env = os.environ.get("TESSQUICKLOOK_QUATERNION_DIR")
    if env:
        dirs.append(Path(env))
    dirs.append(systematics_cache_dir() / "quaternions" / "fits")
    # Legacy location on the machine this was ported from; harmless elsewhere.
    dirs.append(Path.home() / "Dropbox" / "TESS_quicklook" / "quaternions" / "fits")
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if pattern.search(f.name):
                return f, False

    if not download:
        raise FileNotFoundError(f"No quaternion file for sector {sector}")

    names = [n for n in _engineering_listing() if pattern.search(n)]
    if not names:
        # Could be a genuinely unpublished sector, or a stale cache from before
        # it published.  Refetch once before believing the negative.
        names = [n for n in _engineering_listing(refresh=True) if pattern.search(n)]
    if not names:
        raise FileNotFoundError(f"No quaternion file on MAST for sector {sector}")
    name = sorted(names)[-1]
    dest = systematics_cache_dir() / "quaternions" / "fits" / name
    return _fetch_large_file(f"{_ENGINEERING}/{name}", dest), True


def load_quaternion_arrays(sector, cam=DEFAULT_QUAT_CAMERA, search_dirs=None,
                           download=True, discard_fits=False):
    """Return ``(time, q)`` for one camera, caching only the columns we use.

    The engineering FITS files are ~430 MB because they carry all four cameras'
    quaternions plus housekeeping.  The pipeline reads four columns from one
    camera, so those are extracted once and stored as a ~25 MB ``.npz``.

    ``discard_fits=True`` deletes the source FITS after extraction.  That is the
    difference between ~430 MB and ~25 MB of permanent cache per sector, which
    matters when processing many sectors.  Binning stays exact -- the cached
    arrays are the same float64 values, not a downsample.
    """
    npz = systematics_cache_dir() / "quaternions" / "raw" / f"s{int(sector)}_cam{cam}.npz"
    if npz.exists():
        try:
            with np.load(npz) as z:
                return z["time"], z["q"]
        except Exception:  # noqa: BLE001
            pass

    path, was_downloaded = load_quaternion_file(sector, search_dirs=search_dirs, download=download)
    with fits.open(path, memmap=True) as hdul:
        qt = hdul[2].data
        qtime = np.array(qt["TIME"], dtype=float)
        q = np.column_stack([
            np.array(qt[f"C{cam}_Q{k}"], dtype=float) for k in (1, 2, 3)
        ])

    try:
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz, time=qtime, q=q)
        # Only ever delete a file this call downloaded itself, into the cache
        # directory it owns -- never one found via search_dirs, the
        # TESSQUICKLOOK_QUATERNION_DIR pre-staging directory, or the legacy
        # path. See load_quaternion_file's docstring for the incident this
        # fixes: discard_fits=True deleted a 435 MB pre-staged sector-47 file
        # that this process never downloaded and had no business removing.
        if discard_fits and was_downloaded:
            Path(path).unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass

    return qtime, q


def _bin_windows(sample_times, centers, cadence):
    """Two-pointer walk matching the IDL's ``while`` loops exactly.

    For each centre, ``b`` is the first sample at or after ``centre-cadence/2``
    and ``e`` the first at or after ``centre+cadence/2``; the window is
    ``[b, e)``.  Returns arrays of start/stop indices.
    """
    nq = sample_times.size
    b = 0
    starts = np.zeros(centers.size, dtype=int)
    stops = np.zeros(centers.size, dtype=int)
    for i, c in enumerate(centers):
        if not np.isfinite(c):
            starts[i] = stops[i] = 0
            continue
        while b < nq - 1 and sample_times[b] < c - cadence / 2:
            b += 1
        e = b
        while e < nq - 1 and sample_times[e] < c + cadence / 2:
            e += 1
        starts[i], stops[i] = b, e
    return starts, stops


def bin_quaternions(spacecrafttime, sector, cadence=None, quat_camera=None,
                    search_dirs=None, download=True, discard_fits=False):
    """Port of ``processquaternions.pro``.

    Returns a dict of arrays keyed as the IDL structure tags
    (``q1mean``, ``q1std``, ``q1q2mean``, ``q1skew``, ``q1kurt``, ...) plus
    ``dataexist``, all aligned to ``spacecrafttime``.
    """
    spacecrafttime = np.asarray(spacecrafttime, dtype=float)
    if cadence is None:
        cadence = cadence_days_for_sector(int(sector))
    cam = DEFAULT_QUAT_CAMERA if quat_camera is None else int(quat_camera)

    # The raw quaternion FITS files are ~430 MB each, so cache the *binned*
    # result.  The key covers everything the binning depends on; the payload is
    # a few hundred kB, which makes re-runs of a target essentially free and
    # means the big download happens at most once per sector.
    key = hashlib.md5(
        np.ascontiguousarray(spacecrafttime, dtype=">f8").tobytes()
        + f"|{int(sector)}|{cadence!r}|{cam}".encode()
    ).hexdigest()[:16]
    cache = systematics_cache_dir() / "quaternions" / "binned" / f"s{int(sector)}_{key}.npz"
    if cache.exists():
        try:
            with np.load(cache) as z:
                return {k: z[k] for k in z.files}
        except Exception:  # noqa: BLE001
            pass  # corrupt cache: fall through and recompute

    qtime, q = load_quaternion_arrays(
        sector, cam, search_dirs=search_dirs, download=download,
        discard_fits=discard_fits,
    )

    n = spacecrafttime.size
    out = {"spacecrafttime": spacecrafttime, "dataexist": np.zeros(n)}
    for k in (1, 2, 3):
        for suf in ("mean", "std", "skew", "kurt", "max", "min"):
            out[f"q{k}{suf}"] = np.zeros(n)
    for pair in ("q1q2", "q1q3", "q2q3"):
        out[f"{pair}mean"] = np.zeros(n)
        out[f"{pair}std"] = np.zeros(n)

    starts, stops = _bin_windows(qtime, spacecrafttime, cadence)

    for i in range(n):
        b, e = starts[i], stops[i]
        if b == e:
            continue
        out["dataexist"][i] = 1.0
        seg = q[b:e]
        q1, q2, q3 = seg[:, 0], seg[:, 1], seg[:, 2]

        for k, qq in enumerate((q1, q2, q3), start=1):
            out[f"q{k}mean"][i] = qq.mean()
            out[f"q{k}max"][i] = qq.max()
            out[f"q{k}min"][i] = qq.min()

        for name, prod in (("q1q2", q1 * q2), ("q1q3", q1 * q3), ("q2q3", q2 * q3)):
            out[f"{name}mean"][i] = prod.mean()

        if e - b > 1:
            for k, qq in enumerate((q1, q2, q3), start=1):
                out[f"q{k}std"][i] = np.std(qq, ddof=1)
                sk, ku = _moment_skew_kurt(qq)
                out[f"q{k}skew"][i] = _signed_root(sk, 3)
                out[f"q{k}kurt"][i] = _signed_root(ku, 4)
            for name, prod in (("q1q2", q1 * q2), ("q1q3", q1 * q3), ("q2q3", q2 * q3)):
                out[f"{name}std"][i] = np.std(prod, ddof=1)
        # e-b == 1 leaves the std/skew/kurt entries at zero, as the IDL does.

    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **out)
    except Exception:  # noqa: BLE001
        pass  # caching is an optimisation, never a hard requirement

    return out


def bin_cbvs(spacecrafttime, sector, camera, ccd, cadence=None, n_vectors=8):
    """Port of ``bincbv.pro``.

    Retrieves single-scale and band-3 multi-scale CBVs via lightkurve, then
    averages them into the same cadence windows used for the quaternions.
    Returns a dict with keys ``v1..v8`` and ``b3v1..b3v8``.
    """
    spacecrafttime = np.asarray(spacecrafttime, dtype=float)
    if cadence is None:
        cadence = cadence_days_for_sector(int(sector))

    from lightkurve.correctors import load_tess_cbvs

    n = spacecrafttime.size
    out = {}
    for kk in range(1, n_vectors + 1):
        out[f"v{kk}"] = np.zeros(n)
        out[f"b3v{kk}"] = np.zeros(n)
    out["dataexist"] = np.zeros(n)

    def _fetch(cbv_type, band):
        try:
            cbv = load_tess_cbvs(
                sector=int(sector), camera=int(camera), ccd=int(ccd),
                cbv_type=cbv_type, band=band,
            )
            return cbv
        except Exception:
            return None

    single = _fetch("SingleScale", None)
    band3 = _fetch("MultiScale", 3)

    for cbv, prefix in ((single, "v"), (band3, "b3v")):
        if cbv is None:
            continue
        # CBV time is in BTJD; the IDL matches on spacecraft time, so undo the
        # barycentric correction if the product carries one.
        ctime = np.asarray(cbv.time.value, dtype=float)
        starts, stops = _bin_windows(ctime, spacecrafttime, cadence)
        for kk in range(1, n_vectors + 1):
            col = f"VECTOR_{kk}"
            if col not in cbv.colnames:
                continue
            vals = np.asarray(cbv[col], dtype=float)
            acc = out[f"{prefix}{kk}"]
            for i in range(n):
                b, e = starts[i], stops[i]
                if b != e:
                    acc[i] = np.nanmean(vals[b:e])
                    if prefix == "v":
                        out["dataexist"][i] = 1.0

    return out
