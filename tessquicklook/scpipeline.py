"""``quicklooktesssc`` -- the same correction applied to SPOC short-cadence data.

Port of ``quicklooksector3.pro`` (driven in production by ``bulkrunsc.pro``).

This is the 2-minute / 20-second sibling of :func:`~tessquicklook.pipeline.quicklooktessffi`.
The simultaneous systematics + variability fit is *identical* -- the same
:mod:`tessquicklook.decorrelate` and :mod:`tessquicklook.spline` code runs
unchanged.  What differs is everything around it:

===========================  ==========================================  ================================
step                         FFI (``quicklooktessffi``)                  short cadence (``quicklooktesssc``)
===========================  ==========================================  ================================
pixels                       TESScut cutout                              SPOC mission light-curve file
photometry                   10 circular + 10 PRF apertures, then pick   none -- ``SAP_FLUX`` as delivered
dilution                     PRF scene model of TIC neighbours           ``CROWDSAP`` header keyword
barycentric time             ephemeris lookup for the star's coords      ``TIME`` is already BTJD
quaternion regressors        std + mean (+ optional skew, kurt)          std + mean only
CBVs                         single-scale (ext 1) **and** band 3 (ext 5)  band 3 only (ext 5)
background regressor         aperture median + robust mean               ``SAP_BKG``, spline-flattened
===========================  ==========================================  ================================

The IDL SC routine has no ``/skew`` or ``/kurt`` keyword at all; they are offered
here because the machinery is shared and they cost nothing, but they default off
to match.
"""

from __future__ import annotations

import warnings

import numpy as np

from .catalog import query_tic
from .corrections import rebin_lightcurve, undilute
from .decorrelate import build_systematics_vectors, quatcorrect_one
from .idlcompat import point_to_point_scatter
from .spline import keplerspline, keplerspline_design
from .spoc import (
    EXPTIME_FAST,
    EXPTIME_SHORT,
    download_spoc_lightcurves,
    load_spoc_lightcurve,
    match_spoc_cbvs,
    normalise_exptime,
)
from .systematics import bin_quaternions

__all__ = ["quicklooktesssc", "flatten_background"]


def flatten_background(t, bkg, ndays=0.1, maxiter=40, growth=1.3):
    """Remove the slow component of ``SAP_BKG``, as ``quicklooksector3`` does.

    The IDL fits a ``keplerspline`` with ``nd=0.1`` and, if any point of the
    result is non-finite, retries with the knot spacing grown by 1.3 each time
    until it succeeds.  A 0.1-day spline over a 2-minute light curve is very
    stiff to fit and this loop fires regularly, so it is reproduced faithfully.
    """
    bkg = np.asarray(bkg, dtype=float)
    finite = np.isfinite(bkg)
    if finite.sum() < 10:
        return np.zeros_like(bkg)

    filled = np.where(finite, bkg, np.median(bkg[finite]))
    for count in range(int(maxiter)):
        model, _, _ = keplerspline(t, filled, ndays=ndays * growth**count)
        flat = filled - model
        if np.all(np.isfinite(flat)):
            return flat
    warnings.warn("Background flattening never converged; using a zero column.")
    return np.zeros_like(bkg)


def _quality_mask(lc, quats=None, allowscatteredlight=False, noexclude=False):
    """Reproduce the IDL's cadence-rejection expression for SPOC files.

    ``(quality EQ 0 OR quality EQ 32768) AND finite(sap_flux) AND finite(time)``
    plus, when quaternions are in play, ``quats.dataexist``.  Bit 15 (32768) is
    "insufficient targets for error correction" -- a PDC-quality complaint that
    says nothing about SAP, which is why it is kept.
    """
    n = lc["t"].size
    if noexclude:
        return np.isfinite(lc["t"]) & np.isfinite(lc["sap"])

    q = lc["quality"]
    finite = np.isfinite(lc["sap"]) & np.isfinite(lc["t"])

    if allowscatteredlight:
        # Straylight (2048) and Straylight2 (4096).  Note the IDL tests for
        # *equality*, not a bitmask, so a cadence flagged 2048 together with
        # anything else (4160 = 64|4096, say) still fails -- and 32768, which
        # the default branch keeps, is dropped here.  This is therefore not a
        # superset of the default mask, it is a different one.  Reproduced as
        # written; `finite` is the one addition, since a NaN flux cannot be fit
        # whatever its quality flag says.
        keep = (((q == 4096) | (q == 2048)) | (q == 0)) & finite
    else:
        keep = ((q == 0) | (q == 32768)) & finite

    # Applied to every cadence, including the readmitted straylight ones -- the
    # IDL scopes this to the `quality eq 0` term only, which lets cadences with
    # no quaternion coverage through with all-zero regressors.  See the matching
    # note in pipeline._quality_mask.
    if quats is not None:
        keep &= np.asarray(quats["dataexist"], dtype=float)[:n] > 0
    return keep


def quicklooktesssc(
    ticid,
    exptime=EXPTIME_SHORT,
    corrndays=0.2,
    ndays=0.5,
    excludesector=None,
    only_sectors=None,
    usecbv=True,
    skew=False,
    kurt=False,
    nomeans=False,
    order=2,
    torder=5,
    variability_basis="spline",
    contamination=True,
    allowscatteredlight=False,
    noexclude=False,
    rebin=False,
    rebin_minutes=30.0,
    solver="normal",
    discard_quaternion_fits=False,
    outfile=None,
    verbose=True,
):
    """Run the quick-look pipeline on SPOC short-cadence data for one TIC ID.

    Parameters
    ----------
    exptime
        ``120`` / ``"short"`` for 2-minute data, ``20`` / ``"fast"`` for 20 s.
    corrndays
        Knot spacing of the variability basis, in days.  Defaults to 0.2 --
        the value ``bulkrunsc.pro`` passes, tighter than the FFI path's 0.3
        because the cadence is finer.
    contamination
        Apply the ``CROWDSAP`` dilution correction ``(f - (1-c))/c``.
    rebin
        Bin the corrected light curve to ``rebin_minutes`` before returning.
        Off by default: the whole point of this path is the faster cadence, and
        the IDL writes its CSV at the native cadence too (it only rebins for
        transit searching).

    Other parameters carry the same meaning as in
    :func:`~tessquicklook.pipeline.quicklooktessffi`.

    Returns
    -------
    dict
        Same schema as the FFI pipeline -- ``t``, ``f``, ``fcor``, ``fcormed``,
        ``fflat``, ``err_photon``, ``err_empirical``, ``cadence_s``, plus a
        ``sectors`` list.  ``f`` here is normalised PDCSAP (carried for
        comparison, as the IDL does); ``fcor`` is the corrected SAP product and
        is the one to fit.
    """
    from .pipeline import SectorResult, write_lightcurve

    exptime = normalise_exptime(exptime)
    excludesector = {int(s) for s in (excludesector or [])}
    only_sectors = {int(s) for s in only_sectors} if only_sectors else None

    star = query_tic(ticid)
    if verbose:
        print(f"TIC {ticid}: RA={star['ra']:.6f} Dec={star['dec']:.6f} "
              f"Tmag={star['tmag']:.3f}")
        print(f"Looking for SPOC {exptime:.0f}s light curves ...")

    want = sorted(only_sectors - excludesector) if only_sectors else None
    paths = download_spoc_lightcurves(ticid, exptime=exptime, sectors=want,
                                      verbose=verbose)
    if not paths:
        raise RuntimeError(
            f"No SPOC {exptime:.0f}s light curves found for TIC {ticid}"
        )

    sectors_out = []
    allt, allf, allfcor, allfcormed = [], [], [], []
    allerr_ph, allerr_emp, allcad = [], [], []

    for path in paths:
        lc = load_spoc_lightcurve(path)
        sec, cam, ccd = lc["sector"], lc["camera"], lc["ccd"]

        if sec in excludesector:
            if verbose:
                print(f"Manually excluding Sector {sec}.")
            continue
        if only_sectors is not None and sec not in only_sectors:
            continue

        if verbose:
            print(f"Processing Sector {sec} (camera {cam}, ccd {ccd}, "
                  f"{lc['exptime']:.0f}s) ...")

        # --- systematics regressors -------------------------------------
        # TIME - TIMECORR is spacecraft time, per cadence and exact; the FFI
        # path has to reconstruct this from an ephemeris.
        try:
            quats = bin_quaternions(
                lc["spacecrafttime"], sec, cadence=lc["cadence_days"],
                discard_fits=discard_quaternion_fits,
            )
        except FileNotFoundError as exc:
            warnings.warn(
                f"Sector {sec}: no quaternion data available yet ({exc}); "
                f"skipping this sector."
            )
            if verbose:
                print(f"  SKIPPED sector {sec}: quaternions not yet published")
            continue
        except OSError as exc:
            # The data exists but could not be reached -- retryable, unlike the
            # unpublished case above.
            warnings.warn(
                f"Sector {sec}: could not retrieve quaternions ({exc}); "
                f"skipping this sector. This is a transient failure -- re-run "
                f"to pick it up."
            )
            if verbose:
                print(f"  SKIPPED sector {sec}: quaternion fetch failed ({exc})")
            continue

        cbvs = None
        if usecbv:
            try:
                cbvs = match_spoc_cbvs(
                    lc["cadenceno"], sec, cam, ccd,
                    fast=np.isclose(exptime, EXPTIME_FAST),
                )
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"CBVs unavailable for sector {sec}: {exc}")

        keep = _quality_mask(lc, quats, allowscatteredlight, noexclude)
        if keep.sum() < 50:
            warnings.warn(f"Sector {sec}: only {int(keep.sum())} good cadences; skipping")
            continue

        t = lc["t"][keep]
        sap = lc["sap"][keep]
        sap_err = lc["sap_err"][keep]
        pdc = lc["pdcsap"][keep]
        bkg = lc["bkg"][keep]

        quats_k = {k: (np.asarray(v)[keep] if np.asarray(v).shape[:1] == keep.shape else v)
                   for k, v in quats.items()}
        cbvs_k = None
        if cbvs is not None:
            cbvs_k = {k: np.asarray(v)[keep] for k, v in cbvs.items()}

        # The IDL centres the CBVs here (it standardises every column), unlike
        # the FFI path which leaves them alone.  Either is fine: the variability
        # basis carries a constant term, so a shifted regressor only moves the
        # systematics model by an offset, which the median renormalisation below
        # removes.  Centred to match this routine.
        vectors, vecnames = build_systematics_vectors(
            quats_k, cbvs=cbvs_k, bg=False, means=not nomeans,
            skew=skew, kurt=kurt, center_cbvs=True,
        )

        # ...and the "med" variant additionally regresses against the
        # spline-flattened background, which is this routine's analogue of the
        # FFI path's aperture median / robust mean.
        flatbg = flatten_background(t, bkg)
        bgvectors = np.column_stack([vectors, flatbg - flatbg.mean()])

        afull = None
        if variability_basis == "spline":
            afull = keplerspline_design(t, ndays=corrndays)
        elif variability_basis != "poly":
            raise ValueError("variability_basis must be 'poly' or 'spline'")

        norm = np.nanmedian(sap)
        fsap = sap / norm

        common = dict(order=order, torder=torder, afull=afull, solver=solver)
        corlc, sysmodel, _, _ = quatcorrect_one(t, fsap, vectors, **common)
        cormedlc, _, _, _ = quatcorrect_one(t, fsap, bgvectors, **common)

        # --- dilution: one header keyword, no scene model ------------------
        crowdsap = lc["crowdsap"] if contamination else 1.0
        if not np.isfinite(crowdsap) or crowdsap <= 0:
            warnings.warn(f"Sector {sec}: unusable CROWDSAP={crowdsap!r}; not diluting")
            crowdsap = 1.0

        corlc = undilute(corlc / np.nanmedian(corlc), crowdsap)
        cormedlc = undilute(cormedlc / np.nanmedian(cormedlc), crowdsap)

        err_photon = sap_err / norm / crowdsap
        emp = point_to_point_scatter(corlc)

        if verbose:
            print(f"  CROWDSAP={crowdsap:.4f}  {vectors.shape[1]} regressors  "
                  f"scatter {emp * 1e6:.0f} ppm, {int(keep.sum())} cadences")

        cad_s = np.full(t.size, lc["exptime"])
        sectors_out.append(SectorResult(
            sector=sec, camera=cam, ccd=ccd, path=str(path),
            exptime=lc["exptime"], crowdsap=crowdsap,
            t=t, f=pdc / np.nanmedian(pdc), fcor=corlc, fcormed=cormedlc,
            fsap=fsap, sysmodel=sysmodel, bkg=bkg, flatbg=flatbg,
            err_photon=err_photon, err_empirical=emp,
            quats=quats_k, cbvs=cbvs_k, vector_names=vecnames,
            xcms=lc["xcms"][keep], ycms=lc["ycms"][keep],
        ))

        if rebin:
            num = rebin_minutes / (lc["exptime"] / 60.0)
            rb = rebin_lightcurve(t, corlc, es=err_photon,
                                  cadence=lc["exptime"] / 60.0, num=num)
            rbraw = rebin_lightcurve(t, pdc / np.nanmedian(pdc),
                                     cadence=lc["exptime"] / 60.0, num=num)
            rbmed = rebin_lightcurve(t, cormedlc,
                                     cadence=lc["exptime"] / 60.0, num=num)
            allt.append(rb["t"])
            allfcor.append(rb["f"])
            allf.append(rbraw["f"])
            allfcormed.append(rbmed["f"])
            allerr_ph.append(rb["e"])
            allerr_emp.append(np.full(rb["t"].size, emp))
            allcad.append(np.full(rb["t"].size, rebin_minutes * 60.0))
        else:
            allt.append(t)
            allf.append(pdc / np.nanmedian(pdc))
            allfcor.append(corlc)
            allfcormed.append(cormedlc)
            allerr_ph.append(err_photon)
            allerr_emp.append(np.full(t.size, emp))
            allcad.append(cad_s)

    if not sectors_out:
        raise RuntimeError("No sectors survived selection")

    t = np.concatenate(allt)
    idx = np.argsort(t)
    t = t[idx]
    f, fcor, fcormed, err_ph, err_emp, cad = (
        np.concatenate(a)[idx]
        for a in (allf, allfcor, allfcormed, allerr_ph, allerr_emp, allcad)
    )

    flat, _, _ = keplerspline(t, fcor, ndays=ndays)
    with np.errstate(invalid="ignore", divide="ignore"):
        fflat = fcor / flat

    result = {
        "ticid": int(ticid),
        "star": star,
        "t": t,
        "f": f,
        "fcor": fcor,
        "fcormed": fcormed,
        "fflat": fflat,
        "spline": flat,
        "err_photon": err_ph,
        "err_empirical": err_emp,
        "cadence_s": cad,
        "sectors": sectors_out,
        "variability_basis": variability_basis,
        "source": f"SPOC {exptime:.0f}s",
    }

    if outfile:
        write_lightcurve(result, outfile)
        if verbose:
            print(f"Wrote {outfile}")
    return result
