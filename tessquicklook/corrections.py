"""Aperture selection, dilution, barycentric time and rebinning.

Ports ``chooseaperturetess.pro``, ``gettesscontamination.pro``,
``tesstimecorrection.pro`` and ``rebintesslightcurve.pro``.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

from .idlcompat import cdpp, fillarr, point_to_point_scatter

__all__ = [
    "choose_aperture",
    "tess_time_correction",
    "aperture_contamination",
    "rebin_lightcurve",
]

LIGHT_SPEED_AU_DAY = 299792458.0 / 149597870700.0 * 24.0 * 3600.0
BJD_OFFSET = 2457000.0

def _ephem_candidates():
    """Where to look for the TESS orbital ephemeris (optional).

    The ephemeris is only needed to recompute barycentric times for the
    *target's* coordinates rather than the cutout centre; without it the
    pipeline falls back to SPOC's own ``TIMECORR``, which differs by well under
    a second for a small cutout.  So this is a refinement, not a requirement.
    """
    out = []
    env = os.environ.get("TESSQUICKLOOK_EPHEMERIS")
    if env:
        out.append(Path(env))
    out.append(
        Path(os.environ.get("TESSQUICKLOOK_CACHE", Path.home() / ".tessquicklook"))
        / "2018-2024_tessephemeris.idl"
    )
    # Legacy location on the machine this was ported from; harmless elsewhere.
    out.append(Path.home() / "Dropbox" / "TESS_quicklook" / "2018-2024_tessephemeris.idl")
    return out


_EPHEM_CACHE = None


def choose_aperture(
    fcirc,
    fpsf,
    use_cdpp=False,
    maxaperrad=None,
    minaperrad=None,
    cdpp_hours=6,
    cadence_days=None,
):
    """Port of ``chooseaperturetess.pro``.

    Scores all ten circular and ten PRF apertures and returns
    ``(use_circular, best_index, scirc, spsf)``.  Ties go to the circular
    aperture, matching the IDL's ``m1 le m2`` test.
    """
    n = fcirc.shape[1]
    scirc = np.empty(n)
    spsf = np.empty(n)
    for i in range(n):
        if use_cdpp:
            scirc[i] = cdpp(fcirc[:, i], cdpp_hours, cadence_days)
            spsf[i] = cdpp(fpsf[:, i], cdpp_hours, cadence_days)
        else:
            scirc[i] = point_to_point_scatter(fcirc[:, i])
            spsf[i] = point_to_point_scatter(fpsf[:, i])

    if maxaperrad is not None:
        scirc[maxaperrad + 1:] = np.inf
        spsf[maxaperrad + 1:] = np.inf
    if minaperrad is not None:
        scirc[:minaperrad] = np.inf
        spsf[:minaperrad] = np.inf

    scirc = np.where(np.isfinite(scirc), scirc, np.inf)
    spsf = np.where(np.isfinite(spsf), spsf, np.inf)

    i1, i2 = int(np.argmin(scirc)), int(np.argmin(spsf))
    if scirc[i1] <= spsf[i2]:
        return True, i1, scirc, spsf
    return False, i2, scirc, spsf


def _load_ephemeris():
    global _EPHEM_CACHE
    if _EPHEM_CACHE is not None:
        return _EPHEM_CACHE
    from scipy.io import readsav

    for p in _ephem_candidates():
        if p.exists():
            d = readsav(str(p))
            _EPHEM_CACHE = (
                np.asarray(d["horizonjdtdb"], dtype=float),
                np.asarray(d["horizonx"], dtype=float),
                np.asarray(d["horizony"], dtype=float),
                np.asarray(d["horizonz"], dtype=float),
            )
            return _EPHEM_CACHE
    _EPHEM_CACHE = None
    return None


def tess_time_correction(spacecrafttime, ra, dec, fallback_time=None):
    """Port of ``tesstimecorrection.pro``.

    Converts spacecraft time to BJD for the *target's* coordinates by dotting
    the TESS orbital position against the line of sight.

    The bundled ephemeris stops at JD 2460406.5 (2024-04-04).  The IDL's
    out-of-range guard is commented out, so it extrapolates silently; here,
    cadences beyond the ephemeris fall back to SPOC's own ``TIMECORR``-based
    time (``fallback_time``) and a warning is raised.
    """
    spacecrafttime = np.asarray(spacecrafttime, dtype=float)
    eph = _load_ephemeris()
    tjd_times = BJD_OFFSET + spacecrafttime

    if eph is None:
        if fallback_time is None:
            raise FileNotFoundError("TESS ephemeris file not found and no fallback given")
        warnings.warn("TESS ephemeris unavailable; using SPOC TIMECORR instead.")
        return np.asarray(fallback_time, dtype=float)

    jd, ex, ey, ez = eph
    in_range = (tjd_times >= jd.min()) & (tjd_times <= jd.max())

    ox = np.interp(tjd_times, jd, ex)
    oy = np.interp(tjd_times, jd, ey)
    oz = np.interp(tjd_times, jd, ez)

    dec_r = np.radians(dec)
    ra_r = np.radians(ra)
    star = np.array([np.cos(dec_r) * np.cos(ra_r),
                     np.cos(dec_r) * np.sin(ra_r),
                     np.sin(dec_r)])

    dtime = (ox * star[0] + oy * star[1] + oz * star[2]) / LIGHT_SPEED_AU_DAY
    bjd = tjd_times + dtime - BJD_OFFSET

    if not in_range.all():
        n_out = int((~in_range).sum())
        if fallback_time is not None:
            warnings.warn(
                f"{n_out} cadences fall outside the bundled TESS ephemeris "
                f"(ends 2024-04-04); using SPOC TIMECORR for those."
            )
            bjd = np.where(in_range, bjd, np.asarray(fallback_time, dtype=float))
        else:
            warnings.warn(
                f"{n_out} cadences fall outside the bundled TESS ephemeris; "
                f"those times are linearly extrapolated and unreliable."
            )
    return bjd


def build_scene_models(apheader, prf, nsamp, shape, target, neighbours):
    """Port of ``getonetesscontamination.pro``.

    Renders the target and every catalogued neighbour as a scaled PRF at its
    WCS position and returns ``(star_model, total_model)``.  Fluxes are relative
    to the target (``10**(0.4*(Tmag_target - Tmag_neighbour))``), so the models
    only ever enter as a ratio.
    """
    from astropy.wcs import WCS

    from .prf import resample_tess_prf

    ny, nx = shape
    xes, yes = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
    wcs = WCS(apheader)

    tx, ty = wcs.all_world2pix(target["ra"], target["dec"], 0)
    star_model = resample_tess_prf(
        xes, yes, prf, float(tx) - xes.mean(), float(ty) - yes.mean(), nsamp=nsamp
    )
    total_model = star_model.copy()

    if neighbours is not None and len(neighbours):
        nra = np.asarray(neighbours["ra"], dtype=float)
        ndec = np.asarray(neighbours["dec"], dtype=float)
        ntmag = np.asarray(neighbours["Tmag"], dtype=float)
        nxp, nyp = wcs.all_world2pix(nra, ndec, 0)
        fluxes = 10.0 ** (0.4 * (float(target["tmag"]) - ntmag))
        for xi, yi, fl in zip(np.atleast_1d(nxp), np.atleast_1d(nyp), fluxes):
            if not (np.isfinite(xi) and np.isfinite(yi) and np.isfinite(fl)):
                continue
            total_model += fl * resample_tess_prf(
                xes, yes, prf, xi - xes.mean(), yi - yes.mean(), nsamp=nsamp
            )

    return star_model, total_model


def aperture_contamination(circmasks, psfmasks, star_model, total_model):
    """Port of the dilution part of ``gettesscontamination.pro``.

    Returns ``(circcontam, psfcontam)``: the fraction of flux in each aperture
    contributed by the target itself.  Light curves are then de-diluted as
    ``(f - (1 - c)) / c``.
    """
    n = circmasks.shape[0]
    circ = np.zeros(n)
    psf = np.zeros(n)
    for i in range(n):
        tot = (circmasks[i] * total_model).sum()
        circ[i] = (circmasks[i] * star_model).sum() / tot if tot > 0 else 1.0
        tot = (psfmasks[i] * total_model).sum()
        psf[i] = (psfmasks[i] * star_model).sum() / tot if tot > 0 else 1.0
    return circ, psf


def undilute(flux, contam):
    """Apply the IDL's dilution correction ``(f - (1-c))/c``."""
    return (np.asarray(flux, dtype=float) - (1.0 - contam)) / contam


def rebin_lightcurve(ts, fs, es=None, qs=None, cadence=2.0, num=15.0, ngood=None):
    """Port of ``rebintesslightcurve.pro`` (the default fast method-2 branch).

    Bins onto a regular grid of width ``cadence*num`` minutes, keeping only
    bins containing more than ``ngood`` samples.  Errors combine as
    ``sqrt(1/sum(1/e^2))``, matching the IDL.
    """
    ts = np.asarray(ts, dtype=float)
    fs = np.asarray(fs, dtype=float)
    if qs is None:
        qs = np.zeros(ts.size)
    qs = np.asarray(qs, dtype=float)
    if ngood is None:
        ngood = int(np.floor(num * 2.0 / 3.0))

    bint = cadence * num  # minutes
    width = bint / 24.0 / 60.0
    ut = fillarr(width, ts.min(), ts.max())

    order = np.argsort(ts)
    st, sf, sq = ts[order], fs[order], qs[order]
    se = np.asarray(es, dtype=float)[order] if es is not None else None

    # Same two-pointer walk as the IDL.
    nq = st.size
    nut = np.zeros(ut.size, dtype=int)
    fut = np.zeros(ut.size)
    qut = np.zeros(ut.size)
    eut = np.zeros(ut.size) if se is not None else None

    b = 0
    for i, c in enumerate(ut):
        while b < nq - 1 and st[b] < c - width / 2:
            b += 1
        e = b
        while e < nq - 1 and st[e] <= c + width / 2:
            e += 1
        if b != e:
            fut[i] = sf[b:e].mean()
            nut[i] = e - b
            qut[i] = sq[b:e].sum()
            if se is not None:
                w = 1.0 / se[b:e] ** 2
                eut[i] = np.sqrt(1.0 / w.sum())

    keep = nut > ngood
    out = {"t": ut[keep], "f": fut[keep], "q": qut[keep], "n": nut[keep]}
    if se is not None:
        out["e"] = eut[keep]
    return out
