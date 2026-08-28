"""TIC queries, TESS-point sector prediction and TESScut cutout retrieval.

Ports ``queryticid.pro``, ``querytesscut.pro`` and the ``tess_stars2px``
subprocess call in ``quicklooktessffi.pro``.

The IDL shells out to ``wget``/``unzip`` and hand-parses MAST's XML; these use
astroquery and the tess-point package directly, and cache cutouts on disk so
repeat runs of the same target are free.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

__all__ = [
    "query_tic",
    "query_tic_cone",
    "observed_sectors",
    "tess_point_known_sectors",
    "get_tesscut",
    "cutout_cache_dir",
]


def cutout_cache_dir() -> Path:
    d = Path(os.environ.get("TESSQUICKLOOK_CACHE", Path.home() / ".tessquicklook"))
    return d / "tesscut"


def _catalog_query(**kwargs):
    from astroquery.mast import Catalogs

    return Catalogs.query_object(catalog="TIC", **kwargs) if "objectname" in kwargs \
        else Catalogs.query_region(catalog="TIC", **kwargs)


def query_tic(ticid):
    """Fetch the TIC row for one TIC ID.

    Returns a dict with ra, dec, tmag, pmra, pmdec, rad, mass, teff and their
    uncertainties -- the fields ``quicklooktessffi`` prints and uses.
    """
    from astroquery.mast import Catalogs

    cat = Catalogs.query_criteria(catalog="TIC", ID=int(ticid))
    if len(cat) == 0:
        raise ValueError(f"TIC {ticid} not found")
    row = cat[0]

    def g(key):
        try:
            v = row[key]
            return float(v) if v is not None and np.isfinite(float(v)) else np.nan
        except Exception:
            return np.nan

    return {
        "id": int(ticid),
        "ra": g("ra"),
        "dec": g("dec"),
        "tmag": g("Tmag"),
        "pmra": g("pmRA"),
        "pmdec": g("pmDEC"),
        "rad": g("rad"),
        "e_rad": g("e_rad"),
        "mass": g("mass"),
        "e_mass": g("e_mass"),
        "teff": g("Teff"),
        "e_teff": g("e_Teff"),
    }


def query_tic_cone(ra, dec, radius_arcsec=150.0):
    """Cone search used for dilution: the target plus its neighbours."""
    import astropy.units as u
    from astroquery.mast import Catalogs

    cat = Catalogs.query_region(
        f"{ra} {dec}", radius=radius_arcsec * u.arcsec, catalog="TIC"
    )
    return cat


# Special campaigns that the tess-point *package* does not carry, but whose
# pointings are published.  The TESS-point Web Tool's backend does know them,
# which is why it returns sector 1751 while the pip package does not.
#
# sector -> (ra, dec, roll) of the spacecraft boresight, J2000 degrees.
# 1751: the dedicated 3I/ATLAS campaign, from TESS DRN DR136 (April 2026) §1.
#       Verified against the Web Tool for TIC 47319867: this reproduces its
#       camera 3 / CCD 1, col 1937.626 vs 1937.628, row 222.158 vs 222.155.
SPECIAL_SECTOR_POINTINGS = {
    1751: (109.4916, 21.8424, 84.3356),
}


def _special_sector_hits(ra, dec, sectors=None):
    """Run tess-point against the published pointings of special campaigns.

    tess-point's ``sectorOverrideFile`` replaces its whole sector table, so this
    runs as a separate pass and the caller merges the result.
    """
    import tempfile

    from tess_stars2px import TESS_Spacecraft_Pointing_Data
    from tess_stars2px import tess_stars2px_function_entry as _entry

    want = SPECIAL_SECTOR_POINTINGS if sectors is None else {
        k: v for k, v in SPECIAL_SECTOR_POINTINGS.items() if k in set(sectors)
    }
    if not want:
        return np.array([], dtype=int), np.array([], dtype=int)

    rows = [f"{s} {p[0]} {p[1]} {p[2]}" for s, p in sorted(want.items())]
    # genfromtxt yields a 0-d array for a single row, which the constructor
    # cannot take a len() of; pad with a sentinel that lands nowhere real.
    rows.append("999999 0.0 0.0 0.0")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write("\n".join(rows) + "\n")
        path = fh.name
    try:
        sc = TESS_Spacecraft_Pointing_Data(sectorOverrideFile=path)
        out = _entry(0, float(ra), float(dec), scInfo=sc)
    except Exception:  # noqa: BLE001
        return np.array([], dtype=int), np.array([], dtype=int)
    finally:
        os.unlink(path)

    secs = np.atleast_1d(np.asarray(out[3]))
    cams = np.atleast_1d(np.asarray(out[4]))
    if secs.size and secs[0] == -1:
        return np.array([], dtype=int), np.array([], dtype=int)
    keep = np.isin(secs, list(want))
    return secs[keep].astype(int), cams[keep].astype(int)


def tess_point_known_sectors():
    """The set of sectors tess-point's pointing table actually covers.

    Needed because tess-point cannot speak to sectors outside its table -- most
    importantly the special campaigns numbered in the 1000s (e.g. sector 1751,
    the 3I/ATLAS pointing).  For those, absence from tess-point's output means
    "unknown", not "not observed", so the veto must be skipped.
    """
    try:
        from tess_stars2px import TESS_Spacecraft_Pointing_Data

        return set(int(s) for s in np.atleast_1d(TESS_Spacecraft_Pointing_Data().sectors))
    except Exception:  # noqa: BLE001
        return set()


def observed_sectors(ra, dec):
    """Which sectors/cameras actually contain this position (tess-point).

    Mirrors the IDL's ``spawn, 'python -m tess_stars2px -c RA DEC'`` and the
    parsing of columns 5 (sector) and 6 (camera).  Returns
    ``(sectors, cameras)`` as integer arrays; empty if never observed.
    """
    from tess_stars2px import tess_stars2px_function_entry

    out = tess_stars2px_function_entry(0, float(ra), float(dec))
    # (outID, outEclipLong, outEclipLat, outSec, outCam, outCcd, colPix, rowPix, scinfo)
    sectors = np.atleast_1d(np.asarray(out[3]))
    cameras = np.atleast_1d(np.asarray(out[4]))
    if sectors.size and sectors[0] == -1:
        sectors = np.array([], dtype=int)
        cameras = np.array([], dtype=int)
    sectors = sectors.astype(int)
    cameras = cameras.astype(int)

    # Fold in special campaigns the package's own table does not cover.
    ssec, scam = _special_sector_hits(ra, dec)
    if ssec.size:
        sectors = np.concatenate([sectors, ssec])
        cameras = np.concatenate([cameras, scam])
    return sectors, cameras


def get_tesscut(ra, dec, xsize=15, ysize=15, sectors=None, product="SPOC", cache=True):
    """Download TESScut FFI cutouts, one FITS per sector.

    Returns a list of paths.  Cached under the cutout cache directory keyed by
    position and size, so re-running a target does not re-download.
    """
    from astroquery.mast import Tesscut
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    outdir = cutout_cache_dir() / f"ra_{ra:.6f}_dec_{dec:.6f}_{ysize}x{xsize}_{product}"
    outdir.mkdir(parents=True, exist_ok=True)
    coord = SkyCoord(ra, dec, unit="deg")

    # Tesscut takes a single sector or None (= all); a list is serialised
    # verbatim into the URL and returns HTTP 500.
    if sectors is None:
        wanted = [None]
    else:
        wanted = [int(s) for s in np.atleast_1d(sectors)]

    for sec in wanted:
        # Cache per sector, so a partially-populated directory still completes.
        if cache and sec is not None and list(outdir.rglob(f"*-s{sec:04d}-*.fits")):
            continue
        kw = {} if sec is None else {"sector": sec}
        try:
            Tesscut.download_cutouts(
                coordinates=coord,
                size=[ysize, xsize] * u.pixel,
                product=product,
                path=str(outdir),
                inflate=True,
                **kw,
            )
        except Exception as exc:  # noqa: BLE001
            import warnings

            warnings.warn(f"TESScut failed for sector {sec}: {exc}")

    return sorted(outdir.rglob("*.fits"))
