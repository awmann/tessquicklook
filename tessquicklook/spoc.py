"""SPOC target-pixel photometry products: discovery, download, and loading.

The FFI path in :mod:`tessquicklook.pipeline` does its own photometry from a
TESScut cutout.  The short-cadence path in :mod:`tessquicklook.scpipeline` does
not: it takes ``SAP_FLUX`` straight out of SPOC's mission light-curve files,
exactly as ``quicklooksector3.pro`` does.  Everything expensive in the FFI path
-- TESScut, PRF fitting, aperture selection, scene modelling for dilution -- is
therefore absent here, and dilution reduces to the ``CROWDSAP`` header keyword.

Two cadences exist:

``120 s`` ("short")
    ``*-s_lc.fits``, available from Sector 1 onward.
``20 s`` ("fast")
    ``*-a_fast-lc.fits``, available from Sector 27 onward for a subset of
    targets.

Cotrending basis vectors are handled here too, because lightkurve's
``load_tess_cbvs`` explicitly supports only the 2-minute files ("For now, this
function will only load 2-minute cadence CBVs").  The 20 s CBVs do exist at
MAST -- ``*-a_fast-cbv.fits``, alongside the 2-minute file in the same
directory -- they are simply absent from the bulk-download script that
lightkurve greps.  :func:`resolve_cbv_url` finds the 2-minute URL the same way
lightkurve does and then swaps the suffix.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits

__all__ = [
    "EXPTIME_FAST",
    "EXPTIME_SHORT",
    "normalise_exptime",
    "search_spoc",
    "available_cadences",
    "download_spoc_lightcurves",
    "load_spoc_lightcurve",
    "resolve_cbv_url",
    "load_cbv_table",
    "match_spoc_cbvs",
    "spoc_cache_dir",
]

EXPTIME_SHORT = 120.0
EXPTIME_FAST = 20.0

_MAST_FILE = "https://mast.stsci.edu/api/v0.1/Download/file?uri="
_CURL_SCRIPTS = "https://archive.stsci.edu/missions/tess/download_scripts/sector"

# Band-3 multi-scale CBVs, the only ones quicklooksector3.pro uses.  In the
# delivered files the HDU order is: 1 single-scale, 2 spike, 3-5 multi-scale
# bands 1-3 -- so the IDL's ``mrdfits(file, 5)`` is band 3.
CBV_BAND3_EXT = 5
CBV_SINGLESCALE_EXT = 1


def spoc_cache_dir() -> Path:
    root = Path(os.environ.get("TESSQUICKLOOK_CACHE", Path.home() / ".tessquicklook"))
    return root / "spoc"


def normalise_exptime(exptime):
    """Accept 20/120, ``"20s"``/``"120s"``, ``"fast"``/``"short"``/``"2min"``."""
    if exptime is None:
        return None
    if isinstance(exptime, str):
        key = exptime.strip().lower()
        if key in ("fast", "20s", "20", "twentysecond"):
            return EXPTIME_FAST
        if key in ("short", "120s", "120", "2min", "2m"):
            return EXPTIME_SHORT
        raise ValueError(f"Unrecognised exptime {exptime!r}")
    return float(exptime)


def search_spoc(ticid, exptime=None, sectors=None):
    """Return the SPOC light-curve products for a TIC ID as an astropy table.

    Columns of interest: ``sequence_number`` (sector), ``exptime``,
    ``productFilename``, ``dataURI``.
    """
    import logging

    import lightkurve as lk

    exptime = normalise_exptime(exptime)
    # lightkurve catches its own SearchError and logs "No data found for
    # target ..." at ERROR.  Here that is a normal answer -- a target with only
    # FFI coverage -- so it is silenced for the duration of the call, and the
    # empty table below is what carries the result.
    log = logging.getLogger("lightkurve")
    level = log.level
    log.setLevel(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sr = lk.search_lightcurve(
                f"TIC {int(ticid)}", mission="TESS", author="SPOC",
                **({"exptime": exptime} if exptime is not None else {}),
            )
    finally:
        log.setLevel(level)
    table = sr.table
    if len(table) == 0:
        return table

    keep = np.ones(len(table), dtype=bool)
    if exptime is not None:
        keep &= np.isclose(np.asarray(table["exptime"], dtype=float), exptime)
    if sectors is not None:
        want = {int(s) for s in sectors}
        keep &= np.array([int(s) in want for s in table["sequence_number"]])
    return table[keep]


def available_cadences(ticid):
    """Map sector -> sorted exposure times (seconds) of available SPOC products.

    Used by the auto-cadence dispatcher to decide, per sector, whether 20 s,
    120 s, or only FFI data exist.
    """
    table = search_spoc(ticid)
    out = {}
    for row in table:
        sec = int(row["sequence_number"])
        out.setdefault(sec, set()).add(float(row["exptime"]))
    return {k: sorted(v) for k, v in sorted(out.items())}


def _download(url, dest):
    """Cached fetch.  Shares the quaternion downloader's timeout/resume logic --
    a 40 MB fast-lc file is just as capable of hanging a run on a dropped
    connection as a 430 MB quaternion file."""
    from .systematics import _fetch_large_file

    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    return _fetch_large_file(url, dest)


def download_spoc_lightcurves(ticid, exptime=EXPTIME_SHORT, sectors=None,
                              cache=True, verbose=False):
    """Fetch the SPOC light-curve FITS files, returning local paths.

    Files are cached under ``~/.tessquicklook/spoc/`` (override with
    ``TESSQUICKLOOK_CACHE``), keyed by their MAST filename, so a second run of
    the same target costs nothing.
    """
    exptime = normalise_exptime(exptime)
    table = search_spoc(ticid, exptime=exptime, sectors=sectors)
    paths = []
    for row in table:
        name = str(row["productFilename"])
        dest = spoc_cache_dir() / f"tic{int(ticid)}" / name
        if not cache and dest.exists():
            dest.unlink()
        if verbose and not dest.exists():
            print(f"  downloading {name}")
        try:
            paths.append(_download(_MAST_FILE + str(row["dataURI"]), dest))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Could not download {name}: {exc}")
    return sorted(paths, key=lambda p: p.name)


def load_spoc_lightcurve(path):
    """Read one SPOC light-curve file into the dict the SC pipeline expects.

    ``TIME`` in these files is already barycentric (BTJD), so -- unlike the FFI
    path -- no ephemeris lookup is needed.  ``TIME - TIMECORR`` recovers
    spacecraft time for the quaternion binning, per-cadence and exactly, which
    is what ``quicklooksector3.pro`` uses.
    """
    with fits.open(path, memmap=False) as hdul:
        h0 = hdul[0].header
        h1 = hdul[1].header
        d = hdul[1].data
        apheader = hdul[2].header if len(hdul) > 2 else {}
        apmask = np.asarray(hdul[2].data) if len(hdul) > 2 else None

    def col(name, default=np.nan):
        if name in d.columns.names:
            return np.array(d[name], dtype=float)
        return np.full(len(d), default, dtype=float)

    t = col("TIME")
    timecorr = col("TIMECORR", 0.0)

    # TIMEDEL is the cadence, in days.  Note that EXPOSURE is *not* what its
    # name suggests -- it is the total on-target time for the whole sector
    # (21.475 d for Sector 11), so using it here would be wrong by a factor of
    # ~15000.  FRAMETIM x NUM_FRM is the cross-check: 2.0 s x 60 = 120 s.
    timedel = h1.get("TIMEDEL")
    if timedel is None or not np.isfinite(float(timedel)):
        frametim, numfrm = h1.get("FRAMETIM"), h1.get("NUM_FRM")
        if frametim and numfrm:
            timedel = float(frametim) * float(numfrm) / 86400.0
        else:
            good = np.isfinite(t)
            timedel = float(np.median(np.diff(t[good]))) if good.sum() > 2 else np.nan
    exptime = float(timedel) * 86400.0

    return {
        "path": str(path),
        "sector": int(h0.get("SECTOR", -1)),
        "camera": int(h0.get("CAMERA", -1)),
        "ccd": int(h0.get("CCD", -1)),
        "crowdsap": float(h1.get("CROWDSAP", 1.0)),
        "flfrcsap": float(h1.get("FLFRCSAP", 1.0)),
        "cadence_days": float(timedel),
        "exptime": exptime,
        "t": t,
        "timecorr": timecorr,
        "spacecrafttime": t - timecorr,
        "cadenceno": col("CADENCENO", -1),
        "quality": np.nan_to_num(col("QUALITY", 0.0)).astype(np.int64),
        "sap": col("SAP_FLUX"),
        "sap_err": col("SAP_FLUX_ERR"),
        "pdcsap": col("PDCSAP_FLUX"),
        "pdcsap_err": col("PDCSAP_FLUX_ERR"),
        "bkg": col("SAP_BKG"),
        "bkg_err": col("SAP_BKG_ERR"),
        "xcms": col("MOM_CENTR1"),
        "ycms": col("MOM_CENTR2"),
        "apmask": apmask,
        "apheader": dict(apheader) if apheader else {},
        "header0": dict(h0),
        "header1": dict(h1),
    }


def _cbv_curl_script(sector):
    url = f"{_CURL_SCRIPTS}/tesscurl_sector_{int(sector)}_cbv.sh"
    dest = spoc_cache_dir() / "cbv_scripts" / f"tesscurl_sector_{int(sector)}_cbv.sh"
    _download(url, dest)
    return dest.read_text(errors="replace")


def resolve_cbv_url(sector, camera, ccd, fast=False):
    """URL of the CBV file for one sector/camera/CCD.

    The 2-minute URL is recovered from MAST's bulk-download curl script (the
    same route lightkurve takes).  For ``fast=True`` the suffix is swapped:
    ``...-s_cbv.fits`` -> ``...-a_fast-cbv.fits``.  The fast files sit in the
    same directory but are *not* listed in the curl script, which is why
    lightkurve cannot find them.
    """
    text = _cbv_curl_script(sector)
    needle = f"s{int(sector):04d}-{int(camera)}-{int(ccd)}-"
    for m in re.finditer(r"https://\S+?_cbv\.fits", text):
        url = m.group(0)
        if needle in url:
            if fast:
                url = url.replace("-s_cbv.fits", "-a_fast-cbv.fits")
            return url
    raise FileNotFoundError(
        f"No CBV file listed for sector {sector} camera {camera} ccd {ccd}"
    )


def load_cbv_table(sector, camera, ccd, fast=False, ext=CBV_BAND3_EXT):
    """Download (once) and read one CBV extension as a FITS record array."""
    url = resolve_cbv_url(sector, camera, ccd, fast=fast)
    dest = spoc_cache_dir() / "cbv" / url.rsplit("/", 1)[-1]
    _download(url, dest)
    with fits.open(dest, memmap=False) as hdul:
        return np.array(hdul[ext].data), dict(hdul[ext].header)


def match_spoc_cbvs(cadenceno, sector, camera, ccd, fast=False, n_vectors=8,
                    ext=CBV_BAND3_EXT):
    """Align CBV vectors onto a light curve's cadences.

    ``quicklooksector3.pro`` appends ``cbv.vector_k`` to the design matrix
    directly, relying on the CBV file having one row per light-curve cadence in
    the same order.  That happens to hold, but matching on ``CADENCENO`` is
    exact regardless, so that is what this does; a mismatch becomes a warning
    and a zero-filled column rather than a silent misalignment.

    Returns a dict keyed ``b3v1``..``b3vN`` (or ``v1``.. for the single-scale
    extension), holding only the vectors the file actually contains.  Band 3
    typically carries three.
    """
    data, _ = load_cbv_table(sector, camera, ccd, fast=fast, ext=ext)
    prefix = "b3v" if ext == CBV_BAND3_EXT else "v"

    cadenceno = np.asarray(cadenceno, dtype=np.int64)
    out = {}
    names = data.dtype.names

    if "CADENCENO" in names:
        cbv_cad = np.asarray(data["CADENCENO"], dtype=np.int64)
        order = np.argsort(cbv_cad)
        pos = np.searchsorted(cbv_cad[order], cadenceno)
        pos = np.clip(pos, 0, cbv_cad.size - 1)
        idx = order[pos]
        hit = cbv_cad[idx] == cadenceno
        if not hit.all():
            warnings.warn(
                f"Sector {sector}: {int((~hit).sum())} of {cadenceno.size} cadences "
                f"have no matching CBV row; those rows are zeroed."
            )
    else:  # pragma: no cover - all delivered files carry CADENCENO
        if len(data) != cadenceno.size:
            raise ValueError("CBV file has no CADENCENO and a different length")
        idx = np.arange(cadenceno.size)
        hit = np.ones(cadenceno.size, dtype=bool)

    for kk in range(1, int(n_vectors) + 1):
        colname = f"VECTOR_{kk}"
        if colname not in names:
            continue
        vals = np.asarray(data[colname], dtype=float)[idx]
        vals[~hit] = 0.0
        out[f"{prefix}{kk}"] = np.nan_to_num(vals)
    return out
