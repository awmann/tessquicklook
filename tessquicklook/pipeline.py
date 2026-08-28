"""``quicklooktessffi`` -- the main entry point.

Port of ``quicklooktessffi.pro``.

Example
-------
The IDL invocation for HIP 67522::

    quicklooktessffi, 166527623L, corrnd=.3, xsize=30, ysize=30, $
        excludesector=102, /usecbv, /skew, /kurt, /nostop

becomes::

    from tessquicklook import quicklooktessffi
    result = quicklooktessffi(166527623, corrndays=0.3, xsize=30, ysize=30,
                              excludesector=[102], usecbv=True,
                              skew=True, kurt=True)
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np

from .catalog import (
    get_tesscut,
    observed_sectors,
    query_tic,
    query_tic_cone,
    tess_point_known_sectors,
)
from .corrections import (
    aperture_contamination,
    build_scene_models,
    choose_aperture,
    rebin_lightcurve,
    tess_time_correction,
    undilute,
)
from .decorrelate import quatcorrect
from .idlcompat import point_to_point_scatter
from .photometry import extract_photometry
from .prf import get_tess_prf
from .spline import keplerspline
from .systematics import bin_cbvs, bin_quaternions, cadence_days_for_sector

__all__ = ["quicklooktessffi", "SectorResult"]

# Hand-maintained bad-time windows from quicklooktessffi.pro.
_BAD_TIME_WINDOWS = [
    (1347.0, 1350.0),
    (1395.4422, 1396.6366),
    (1420.0, 1424.0),
    (1530.0, 1532.0),
]
_BAD_FFI_WINDOWS = [
    (2020016175923, 2020016225923),
    (2020043182922, 2020043232922),
]
_BAD_FFI_EXACT = {2018358115937}


class SectorResult(dict):
    """Per-sector products, addressable as a dict."""


def _sector_from_filename(path):
    """TESScut names look like tess-s0011-1-2_..._astrocut.fits."""
    m = re.search(r"[-_]s(\d{4})[-_](\d)[-_](\d)", Path(path).name)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None, None, None


def _quality_mask(rawdata, allow_scattered_light=False, mediancut=None,
                  quats=None, noexclude=False):
    """Reproduce the IDL's cadence-rejection expression."""
    n = rawdata["t"].size
    if noexclude:
        return np.ones(n, dtype=bool)

    q = rawdata["quality"]
    t = rawdata["t"]
    sector = rawdata["sector"]

    if allow_scattered_light:
        keep = (q == 4096) | (q == 2048)
        base = q == 0
        for lo, hi in _BAD_TIME_WINDOWS[:3]:
            base &= (t < lo) | (t > hi)
        keep = keep | base
    else:
        keep = (q == 0) | ((sector == 26) & (q == 2048))
        for lo, hi in _BAD_TIME_WINDOWS:
            keep &= (t < lo) | (t > hi)

        ffi = rawdata.get("ffidatetime")
        if ffi is not None:
            ffinum = np.array([int(s) if str(s).isdigit() else -1 for s in ffi])
            for lo, hi in _BAD_FFI_WINDOWS:
                keep &= (ffinum < lo) | (ffinum > hi)
            for bad in _BAD_FFI_EXACT:
                keep &= ffinum != bad

    # DELIBERATE DEVIATION from the IDL.  Both IDL routines apply the
    # `quats.dataexist` requirement only inside the *non*-scattered branch, so
    # `/allowscattered` silently admits cadences with no quaternion coverage.
    # Those rows enter the design matrix with every quaternion regressor set to
    # zero -- and zero is not neutral in a least-squares fit: it drags the
    # column means used for centring and pulls the fit toward the intercept.
    # Measured across the cached SPOC files, 35670 of 543747 straylight-flagged
    # cadences (6.6%) lack quaternion coverage, reaching 45% in TIC 410214986
    # sector 68.  Applied unconditionally here.
    if quats is not None:
        keep &= np.asarray(quats["dataexist"], dtype=float) > 0

    if mediancut is not None:
        keep &= rawdata["medians"] < mediancut

    return keep


def _subset(d, mask, keys):
    out = dict(d)
    for k in keys:
        v = d.get(k)
        if v is None:
            continue
        v = np.asarray(v)
        if v.shape and v.shape[0] == mask.size:
            out[k] = v[mask]
    return out


def quicklooktessffi(
    ticid,
    xsize=15,
    ysize=15,
    corrndays=0.3,
    ndays=0.5,
    excludesector=None,
    only_sectors=None,
    usecbv=False,
    skew=False,
    kurt=False,
    nomeans=False,
    torder=5,
    order=2,
    srin=0.0,
    nosmallmasks=False,
    nomedsmallmask=False,
    usebiggerpsf=False,
    offcollateral=False,
    noexclude=False,
    allowscatteredlight=False,
    mediancut=None,
    maxaperrad=None,
    minaperrad=None,
    ousecirc=None,
    obest=None,
    variability_basis="spline",
    contamination=True,
    contam_radius=150.0,
    use_cdpp=True,
    idl_cdpp_window=False,
    rebin=True,
    product="SPOC",
    solver="normal",
    discard_quaternion_fits=False,
    outfile=None,
    verbose=True,
):
    """Run the quick-look FFI pipeline for one TIC ID.

    Parameters mirror the IDL keywords.  ``variability_basis`` is the one
    addition, and it matters:

    ``"spline"`` (default)
        The flexible B-spline basis the IDL was written to use, controlled by
        ``corrndays``.  Validated against the reference light curves: for
        TIC 88785435 sector 11 it gives 890 ppm point-to-point against the
        IDL's 892 ppm.

    ``"poly"``
        The degree-``torder`` polynomial fallback that ``decorrelatehr858``
        takes whenever ``afull`` is undefined.  A polynomial cannot track a
        young star's rotation across a sector, so the quaternion vectors absorb
        the stellar signal and it is then subtracted -- the same target
        degrades to 2703 ppm.  Provided for diagnosis, not for science.

    Returns a dict with the stitched light curve (``t``, ``f``, ``fcor``,
    ``fcormed``, ``err_photon``, ``err_empirical``) and per-sector detail.
    """
    excludesector = set(int(s) for s in (excludesector or []))
    only_sectors = set(int(s) for s in only_sectors) if only_sectors else None

    star = query_tic(ticid)
    if not np.isfinite(star["ra"]):
        raise ValueError(f"TIC {ticid} has no usable position")
    if verbose:
        print(f"TIC {ticid}: RA={star['ra']:.6f} Dec={star['dec']:.6f} "
              f"Tmag={star['tmag']:.3f}")

    # --- which sectors is the star actually on? ---------------------------
    known_sectors = tess_point_known_sectors()
    if offcollateral:
        obs_sectors, obs_cameras = None, None
    else:
        obs_sectors, obs_cameras = observed_sectors(star["ra"], star["dec"])
        if verbose:
            print(f"tess-point predicts {len(set(obs_sectors.tolist()))} sectors")

    # Fetch exactly the sectors we intend to use, so the on-disk cache can be
    # completed incrementally rather than all-or-nothing.
    if only_sectors:
        fetch = sorted(only_sectors)
    elif obs_sectors is not None and obs_sectors.size:
        # Ask MAST directly as well, so special campaigns outside tess-point's
        # table (e.g. sector 1751) are still fetched.
        wanted = set(obs_sectors.tolist())
        try:
            from astropy.coordinates import SkyCoord
            from astroquery.mast import Tesscut

            avail = Tesscut.get_sectors(
                coordinates=SkyCoord(star["ra"], star["dec"], unit="deg"),
                product=product,
            )
            offered = {int(x) for x in avail["sector"]}
            wanted |= {s for s in offered if s not in known_sectors}
            # TESScut will serve a cutout even where the target lands on
            # collateral/off-detector pixels; tess-point vetoes those.  Say so
            # rather than dropping the sector silently.
            vetoed = sorted(offered - wanted)
            if vetoed and verbose:
                print(f"TESScut offers sector(s) {vetoed} but tess-point places "
                      f"the target off-detector there; skipping "
                      f"(pass offcollateral=True to keep them).")
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Could not list TESScut sectors: {exc}")
        fetch = sorted(wanted - excludesector)
    else:
        fetch = None
    files = get_tesscut(star["ra"], star["dec"], xsize=xsize, ysize=ysize,
                        sectors=fetch, product=product)
    if not files:
        raise RuntimeError("TESScut returned no cutouts")

    # Aperture ladder assumes bleed-trail behaviour keyed on Tmag+4.5.
    eqtmag = star["tmag"] + 4.5

    sectors_out = []
    allt, allf, allfcor, allfcormed = [], [], [], []
    allerr_ph, allerr_emp, allcad = [], [], []

    for path in files:
        sec, cam, ccd = _sector_from_filename(path)
        if sec is None:
            warnings.warn(f"Could not parse sector from {path}; skipping")
            continue

        if only_sectors is not None and sec not in only_sectors:
            continue

        if sec in excludesector:
            if verbose:
                print(f"Manually excluding Sector {sec}.")
            continue

        if obs_sectors is not None:
            on = np.any((obs_sectors == sec) & (obs_cameras == cam))
            if not on:
                # tess-point can only veto sectors its pointing table covers.
                # Special campaigns numbered in the 1000s (sector 1751, the
                # 3I/ATLAS pointing) are absent from it entirely; there,
                # TESScut having returned a cutout is itself the evidence that
                # the star landed on silicon.
                if sec not in known_sectors:
                    if verbose:
                        print(f"Sector {sec} is outside tess-point's table "
                              f"(covers {min(known_sectors)}-{max(known_sectors)}); "
                              f"trusting TESScut and keeping it.")
                else:
                    if verbose:
                        print(f"Automatically excluding Sector {sec} Camera {cam} "
                              f"based on TESS-point information.")
                    continue

        if verbose:
            print(f"Processing Sector {sec} (camera {cam}, ccd {ccd}) ...")

        raw = extract_photometry(
            path, star["ra"], star["dec"], kp=eqtmag, srin=srin,
            smallmasks=not nosmallmasks, medsmallmask=not nomedsmallmask,
            usebiggerpsf=usebiggerpsf,
        )

        cadence = cadence_days_for_sector(sec)
        # Quaternion engineering files lag the science data by a sector or two.
        # Without them there are no systematics regressors at all, so skip the
        # sector rather than abort the target -- the other sectors are fine.
        try:
            quats = bin_quaternions(raw["spacecrafttime"], sec, cadence=cadence,
                                    discard_fits=discard_quaternion_fits)
        except FileNotFoundError as exc:
            warnings.warn(
                f"Sector {sec}: no quaternion data available yet ({exc}); "
                f"skipping this sector."
            )
            if verbose:
                print(f"  SKIPPED sector {sec}: quaternions not yet published")
            continue
        except OSError as exc:
            # Distinct from the above: the data exists, we could not reach it.
            # Worth separating because it is retryable -- and because catching
            # only FileNotFoundError meant a MAST outage took out the entire
            # branch rather than the one sector that needed the fetch.
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
                cbvs = bin_cbvs(raw["spacecrafttime"], sec, cam, ccd, cadence=cadence)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"CBVs unavailable for sector {sec}: {exc}")

        keep = _quality_mask(raw, allowscatteredlight, mediancut, quats, noexclude)
        if keep.sum() < 50:
            warnings.warn(f"Sector {sec}: only {int(keep.sum())} good cadences; skipping")
            continue

        vec_keys = ["t", "spacecrafttime", "timecorr", "quality", "cadenceno",
                    "fcirc", "fpsf", "fcircr", "fpsfr", "err_circ_photon",
                    "err_psf_photon", "medians", "robmeans", "xcms", "ycms",
                    "ffidatetime"]
        raw_k = _subset(raw, keep, vec_keys)
        quats_k = {k: (np.asarray(v)[keep] if np.asarray(v).shape[:1] == keep.shape else v)
                   for k, v in quats.items()}
        cbvs_k = None
        if cbvs is not None:
            cbvs_k = {k: (np.asarray(v)[keep] if np.asarray(v).shape[:1] == keep.shape else v)
                      for k, v in cbvs.items()}

        common = dict(order=order, torder=torder, means=not nomeans, skew=skew,
                      kurt=kurt, variability_basis=variability_basis,
                      corrndays=corrndays, solver=solver)
        data = quatcorrect(raw_k, quats_k, cbvs=cbvs_k, bg=False, **common)
        datamed = quatcorrect(raw_k, quats_k, cbvs=cbvs_k, bg=True, **common)

        # --- dilution, as tesstimeandcontamcorrect does before choosing ------
        circcontam = np.ones(raw_k["fcirc"].shape[1])
        psfcontam = np.ones(raw_k["fpsf"].shape[1])
        if contamination:
            try:
                neighbours = query_tic_cone(star["ra"], star["dec"], contam_radius)
                if neighbours is not None and len(neighbours):
                    neighbours = neighbours[
                        np.asarray(neighbours["ID"]).astype(str) != str(int(ticid))
                    ]
                prf, _, _, nsamp = get_tess_prf(
                    cam, ccd,
                    rowcenter=raw["ext1header"].get("2CRV4P", 0) + raw["timage"].shape[0] / 2.0,
                    colcenter=raw["ext1header"].get("1CRV4P", 0) + raw["timage"].shape[1] / 2.0,
                    sector=sec,
                )
                star_model, total_model = build_scene_models(
                    raw["apheader"], prf, nsamp, raw["timage"].shape, star, neighbours
                )
                circcontam, psfcontam = aperture_contamination(
                    raw["circmasks"], raw["psfmasks"], star_model, total_model
                )
                if verbose:
                    print(f"  dilution: target contributes "
                          f"{circcontam.min():.3f}-{circcontam.max():.3f} of aperture flux")
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"Dilution correction failed for sector {sec}: {exc}")

        for arrname, contam in (("fcirc", circcontam), ("fpsf", psfcontam)):
            for holder in (raw_k, data, datamed):
                for suffix in ("", "cor", "corflat"):
                    key = arrname + suffix
                    if key in holder and np.asarray(holder[key]).ndim == 2:
                        holder[key] = undilute(holder[key], contam[None, :])
            key = "err_circ_photon" if arrname == "fcirc" else "err_psf_photon"
            raw_k[key] = raw_k[key] / contam[None, :]

        usecirc, best, scirc, spsf = choose_aperture(
            data["fcirccor"], data["fpsfcor"], use_cdpp=use_cdpp,
            maxaperrad=maxaperrad, minaperrad=minaperrad,
            cadence_days=None if idl_cdpp_window else cadence,
        )
        if ousecirc is not None:
            usecirc = bool(ousecirc)
        if obest is not None:
            best = int(obest)

        kind = "fcirc" if usecirc else "fpsf"
        errkey = "err_circ_photon" if usecirc else "err_psf_photon"

        thisf = raw_k[kind][:, best]
        thisfcor = data[kind + "cor"][:, best]
        thisfcormed = datamed[kind + "cor"][:, best]
        thiserr = raw_k[errkey][:, best]

        # Barycentric time for the star's own coordinates.
        tcor = tess_time_correction(
            raw_k["spacecrafttime"], star["ra"], star["dec"],
            fallback_time=raw_k["t"],
        )

        emp = point_to_point_scatter(thisfcor)

        if verbose:
            print(f"  aperture: {'circular' if usecirc else 'PRF'} #{best}, "
                  f"scatter {emp * 1e6:.0f} ppm, {keep.sum()} cadences")

        sectors_out.append(SectorResult(
            sector=sec, camera=cam, ccd=ccd, path=str(path),
            exptime=cadence * 86400.0,
            t=tcor, f=thisf, fcor=thisfcor, fcormed=thisfcormed,
            err_photon=thiserr, err_empirical=emp,
            usecirc=usecirc, best=best, scirc=scirc, spsf=spsf,
            raw=raw_k, data=data, datamed=datamed,
            quats=quats_k, cbvs=cbvs_k,
            circmasks=raw["circmasks"], psfmasks=raw["psfmasks"],
            timage=raw["timage"], prf_fit=raw["prf_fit"],
        ))

        # --- rebin the newer, faster cadences onto ~30 min, as the IDL does --
        if rebin and sec >= 27:
            num, cad = (3.0, 10.0) if sec < 56 else (9.0, 200.0 / 60.0)
            rb = rebin_lightcurve(tcor, thisfcor, es=thiserr, cadence=cad, num=num)
            rbraw = rebin_lightcurve(tcor, thisf, cadence=cad, num=num)
            rbmed = rebin_lightcurve(tcor, thisfcormed, cadence=cad, num=num)
            allt.append(rb["t"])
            allfcor.append(rb["f"])
            allf.append(rbraw["f"])
            allfcormed.append(rbmed["f"])
            allerr_ph.append(rb["e"])
            allerr_emp.append(np.full(rb["t"].size, emp))
            allcad.append(np.full(rb["t"].size, cad * num * 60.0))
        else:
            allt.append(tcor)
            allf.append(thisf)
            allfcor.append(thisfcor)
            allfcormed.append(thisfcormed)
            allerr_ph.append(thiserr)
            allerr_emp.append(np.full(tcor.size, emp))
            allcad.append(np.full(tcor.size, cadence * 86400.0))

    if not sectors_out:
        raise RuntimeError("No sectors survived selection")

    t = np.concatenate(allt)
    f = np.concatenate(allf)
    fcor = np.concatenate(allfcor)
    fcormed = np.concatenate(allfcormed)
    err_ph = np.concatenate(allerr_ph)
    err_emp = np.concatenate(allerr_emp)
    cad_s = np.concatenate(allcad)

    order_idx = np.argsort(t)
    t, f, fcor, fcormed, err_ph, err_emp, cad_s = (
        a[order_idx] for a in (t, f, fcor, fcormed, err_ph, err_emp, cad_s)
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
        "cadence_s": cad_s,
        "sectors": sectors_out,
        "variability_basis": variability_basis,
        "source": "FFI",
    }

    if outfile:
        write_lightcurve(result, outfile)
        if verbose:
            print(f"Wrote {outfile}")

    return result


def write_lightcurve(result, outfile):
    """Write the stitched light curve, including both uncertainty columns.

    The trailing ``cadence_s`` column records each point's exposure time.  It
    matters once a target mixes data sources -- an ``"auto"`` run can hand back
    20 s SPOC data for one sector and 200 s FFI data for the next.
    """
    import csv

    n = np.size(result["t"])
    cadence = result.get("cadence_s")
    if cadence is None:
        cadence = np.full(n, np.nan)

    rows = zip(
        result["t"], result["fcor"], result["fcormed"], result["f"],
        result["fflat"], result["err_photon"], result["err_empirical"],
        cadence,
    )
    with open(outfile, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "flux", "flux_med", "flux_raw", "flux_flat",
                    "flux_err_photon", "flux_err_empirical", "cadence_s"])
        for row in rows:
            w.writerow([f"{v:.10f}" if np.isfinite(v) else "" for v in row])
