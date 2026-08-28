"""Aperture photometry from TESScut FFI cutouts.

Port of ``extractphotometrytess.pro``.

Builds the same twenty apertures the IDL does -- ten circular, ten shaped by
PRF contours -- measures a local background per cadence, sums flux in every
aperture, and tracks centroids.

Uncertainties
-------------
The IDL emits no per-point errors.  Two independent estimates are added here
and carried through to the output:

``flux_err_photon``
    Propagated from SPOC's own per-pixel ``FLUX_ERR`` extension summed in
    quadrature over the aperture, plus the variance of the local background
    estimate scaled by aperture area.  Falls back to Poisson statistics on
    ``RAW_CNTS`` when ``FLUX_ERR`` is unavailable.

``flux_err_empirical``
    The point-to-point scatter ``1.48 * MAD / sqrt(2)`` of the finished light
    curve -- the same statistic ``chooseaperturetess`` already uses to rank
    apertures.  Constant per sector, but it captures scatter the photon budget
    misses (pointing jitter, residual systematics).
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from .idlcompat import contiguous_region, logspace_idl, point_to_point_scatter
from .prf import fit_tess_prf, get_tess_prf, resample_tess_prf

__all__ = ["extract_photometry", "N_MASK_LEVELS"]

N_MASK_LEVELS = 10
BIG_MASK_CUTOFF = 9.5  # Tmag+4.5 below which bleed trails force a different mask


def _aperture_levels(kp, smallmasks=True, usebiggerpsf=False):
    """Reproduce the circular radii and PRF contour levels of the IDL."""
    circ = logspace_idl(1.5, 13, N_MASK_LEVELS)
    psf = logspace_idl(5e-6, 5e-2, N_MASK_LEVELS)[::-1]

    if kp < BIG_MASK_CUTOFF:
        psf = logspace_idl(0.002, 0.1, N_MASK_LEVELS)[::-1]
        if usebiggerpsf:
            psf = logspace_idl(0.0002, 0.02, N_MASK_LEVELS)[::-1]

    if smallmasks:
        # Changed from logspace(0.6,5,10) on 2019-10-25 per the IDL comment.
        circ = logspace_idl(0.4, 5, N_MASK_LEVELS)
        psf = logspace_idl(5e-4, 5e-1, N_MASK_LEVELS)[::-1]

    return circ, psf


def _background_mask_index(smallmasks=True, medsmallmask=True):
    """Which circular aperture defines 'inside the star' for background stats."""
    if smallmasks:
        return 7 if medsmallmask else 8
    return 3 if medsmallmask else 5


def extract_photometry(
    filename,
    ra,
    dec,
    kp=30.0,
    srin=0.0,
    xoff=None,
    yoff=None,
    smallmasks=True,
    medsmallmask=True,
    usebiggerpsf=False,
    nofit=True,
    nomedians=False,
    addback=False,
):
    """Port of ``extractphotometrytess``.

    Returns a dict of per-cadence arrays.  ``fcirc``/``fpsf`` have shape
    ``(ncadence, 10)`` and are normalised to a median of 1 over good cadences,
    matching the IDL.
    """
    with fits.open(filename) as hdul:
        header = hdul[0].header
        tab = hdul[1].data
        ext1header = hdul[1].header
        apheader = hdul[2].header

    camera = int(header.get("CAMERA", 0))
    ccd = int(header.get("CCD", 0))
    sector = int(header.get("SECTOR", 0))

    time = np.asarray(tab["TIME"], dtype=float)
    timecorr = np.asarray(tab["TIMECORR"], dtype=float)
    quality = np.asarray(tab["QUALITY"], dtype=int)
    cadenceno = np.asarray(tab["CADENCENO"], dtype=float)
    flux_cube = np.asarray(tab["FLUX"], dtype=float)  # (ncad, ny, nx)
    raw_cube = np.asarray(tab["RAW_CNTS"], dtype=float)

    colnames = {c.upper() for c in tab.columns.names}
    bkg_cube = np.asarray(tab["FLUX_BKG"], dtype=float) if "FLUX_BKG" in colnames else None
    err_cube = np.asarray(tab["FLUX_ERR"], dtype=float) if "FLUX_ERR" in colnames else None
    ffi_file = np.asarray(tab["FFI_FILE"]) if "FFI_FILE" in colnames else None

    ncad, ny, nx = flux_cube.shape

    if addback and bkg_cube is not None:
        flux_cube = flux_cube + np.nan_to_num(bkg_cube)

    flux_cube = np.nan_to_num(flux_cube, nan=0.0, posinf=0.0, neginf=0.0)
    raw_cube = np.nan_to_num(raw_cube, nan=0.0, posinf=0.0, neginf=0.0)

    # --- pixel grids ------------------------------------------------------
    xes, yes = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))

    xoff = 0.0 if xoff is None else float(xoff)
    yoff = 0.0 if yoff is None else float(yoff)

    # --- locate the star via the cutout WCS (IDL: adxy on the aperture HDU) --
    wcs = WCS(apheader)
    xcenter, ycenter = wcs.all_world2pix(ra, dec, 0)
    xcenter, ycenter = float(xcenter), float(ycenter)

    astrometry_ok = (0 < xcenter < xes.max()) and (0 < ycenter < yes.max())
    if astrometry_ok:
        res = np.hypot(xes - xcenter - xoff, yes - ycenter - yoff)
    else:
        res = np.hypot(xes - xoff, yes - yoff)

    searchradius = (res <= srin).astype(float)
    if srin == 0:
        # IDL: res le 0 selects only exact-zero pixels; with the default offsets
        # that is empty, so the brightest-pixel search falls back to the TIC
        # position below.
        searchradius = np.zeros_like(res)

    # --- reference image: median over time --------------------------------
    lastone = np.median(flux_cube, axis=0)
    rlastone = np.median(raw_cube, axis=0)
    lastone = np.nan_to_num(lastone)
    rlastone = np.nan_to_num(rlastone)

    if searchradius.sum() > 0 and (lastone * searchradius).sum() != 0:
        idx = np.argmax(lastone * searchradius)
        ym, xm = np.unravel_index(idx, lastone.shape)
        xm, ym = float(xm), float(ym)
    elif np.isfinite(xcenter) and np.isfinite(ycenter):
        xm, ym = xcenter, ycenter
    else:
        xm, ym = xoff, yoff

    # --- fit the PRF to the reference image -------------------------------
    prf, _, _, nsamp = get_tess_prf(
        camera,
        ccd,
        rowcenter=ext1header.get("2CRV4P", 0) + ny / 2.0,
        colcenter=ext1header.get("1CRV4P", 0) + nx / 2.0,
        sector=sector,
    )

    inrad = res < 3
    p0 = np.array(
        [
            (lastone[inrad].max() - np.median(lastone)) * 5.0,
            xm - xes.mean(),
            ym - yes.mean(),
            np.median(lastone),
        ]
    )
    pfit = fit_tess_prf(
        xes[inrad], yes[inrad], lastone[inrad], prf, p0,
        fit_center=(srin != 0), nsamp=nsamp,
    )

    # --- circular apertures, centred on the brightest pixel ---------------
    r = np.hypot(xes - xm, yes - ym)
    circlevels, psflevels = _aperture_levels(kp, smallmasks, usebiggerpsf)

    circmasks = np.zeros((N_MASK_LEVELS, ny, nx))
    for j, lev in enumerate(circlevels):
        m = (r <= lev).astype(float)
        if m.sum() == 0:
            m = (r == r.min()).astype(float)
        circmasks[j] = m

    # --- PRF-shaped apertures ---------------------------------------------
    psfmasks = np.zeros_like(circmasks)
    if kp > BIG_MASK_CUTOFF:
        p_nodc = pfit.copy()
        p_nodc[3] = 0.0
        model = p_nodc[0] * resample_tess_prf(
            xes, yes, prf, p_nodc[1], p_nodc[2], nsamp=nsamp
        )
        tmodel = model.sum()
        if tmodel <= 0 or not np.isfinite(tmodel) or (model / tmodel >= 5e-5).sum() <= 0:
            psfmasks = circmasks.copy()  # IDL: 'PSF Fitting Failed'
        else:
            norm = model / tmodel
            for j, lev in enumerate(psflevels):
                m = (norm >= lev).astype(float)
                if m.sum() == 0:
                    m = np.zeros_like(norm)
                    m.flat[np.argmax(model)] = 1.0
                psfmasks[j] = m
    else:
        # Bright star with bleed trails: use contiguous regions of the stacked
        # image instead of the PRF model.
        timage2 = flux_cube.sum(axis=0)
        timage2 = np.nan_to_num(timage2)
        med2 = np.median(timage2)
        timage2 = np.where(timage2 == 0, med2, timage2)
        denom = timage2.max() - med2
        for j, lev in enumerate(psflevels):
            binary = ((timage2 - med2) / denom) > lev
            psfmasks[j] = contiguous_region(binary, xm, ym)

    bkg_idx = _background_mask_index(smallmasks, medsmallmask)
    bkgmask = circmasks[bkg_idx]
    notmask = bkgmask < 1
    n_bkg = int(notmask.sum())

    # --- per-cadence measurement ------------------------------------------
    fcirc = np.zeros((ncad, N_MASK_LEVELS))
    fpsf = np.zeros((ncad, N_MASK_LEVELS))
    fcircr = np.zeros((ncad, N_MASK_LEVELS))
    fpsfr = np.zeros((ncad, N_MASK_LEVELS))
    var_circ = np.zeros((ncad, N_MASK_LEVELS))
    var_psf = np.zeros((ncad, N_MASK_LEVELS))
    medians = np.zeros(ncad)
    robmeans = np.zeros(ncad)
    xcms = np.zeros(ncad)
    ycms = np.zeros(ncad)

    circ_area = circmasks.sum(axis=(1, 2))
    psf_area = psfmasks.sum(axis=(1, 2))

    for j in range(ncad):
        image = flux_cube[j]
        rimage = raw_cube[j]

        bkgpix = image[notmask]
        med = np.median(bkgpix)
        medians[j] = med
        mederror = np.median(np.abs(bkgpix - med))
        sel = np.abs(bkgpix - med) < 2 * mederror
        robmeans[j] = bkgpix[sel].mean() if sel.any() else med

        sub = image if nomedians else (image - med)
        rsub = rimage - np.median(rimage[notmask])

        for i in range(N_MASK_LEVELS):
            fcirc[j, i] = (sub * circmasks[i]).sum()
            fpsf[j, i] = (sub * psfmasks[i]).sum()
            fcircr[j, i] = (rsub * circmasks[i]).sum()
            fpsfr[j, i] = (rsub * psfmasks[i]).sum()

        if err_cube is not None:
            e2 = np.nan_to_num(err_cube[j]) ** 2
        else:
            # Poisson fallback on raw counts, converted back to flux units.
            scale = np.where(rimage > 0, np.abs(image) / np.maximum(rimage, 1e-12), 0.0)
            e2 = np.maximum(rimage, 0.0) * scale**2

        # Uncertainty on the background *level*.  This must come from the
        # per-pixel photon errors, not from the scatter of background pixel
        # values: the latter is dominated by real field structure (neighbouring
        # stars, the target's own wings) and overestimates the noise by orders
        # of magnitude.  The median of n_bkg pixels has variance ~ (pi/2)
        # sigma^2 / n_bkg.
        var_medest = (np.pi / 2.0) * float(e2[notmask].mean()) / max(n_bkg, 1)

        for i in range(N_MASK_LEVELS):
            var_circ[j, i] = (e2 * circmasks[i]).sum() + circ_area[i] ** 2 * var_medest
            var_psf[j, i] = (e2 * psfmasks[i]).sum() + psf_area[i] ** 2 * var_medest

        cms = (sub * bkgmask)
        tot = cms.sum()
        if tot != 0:
            xcms[j] = (cms * xes).sum() / tot
            ycms[j] = (cms * yes).sum() / tot
        else:
            xcms[j] = ycms[j] = np.nan

    # --- normalise each aperture by its median over good cadences ---------
    after = quality == 0
    if not after.any():
        after = np.ones(ncad, dtype=bool)

    err_circ = np.sqrt(var_circ)
    err_psf = np.sqrt(var_psf)
    for i in range(N_MASK_LEVELS):
        for arr, err in ((fcirc, err_circ), (fpsf, err_psf)):
            norm = np.median(arr[after, i])
            if norm != 0 and np.isfinite(norm):
                arr[:, i] /= norm
                err[:, i] /= abs(norm)
        for arr in (fcircr, fpsfr):
            norm = np.median(arr[after, i])
            if norm != 0 and np.isfinite(norm):
                arr[:, i] /= norm

    # Empirical scatter, one number per aperture.
    emp_circ = np.array([point_to_point_scatter(fcirc[after, i]) for i in range(N_MASK_LEVELS)])
    emp_psf = np.array([point_to_point_scatter(fpsf[after, i]) for i in range(N_MASK_LEVELS)])

    timage = flux_cube[quality == 0].sum(axis=0) if (quality == 0).any() else flux_cube.sum(axis=0)

    ffidatetime = None
    if ffi_file is not None:
        ffidatetime = np.array(
            [str(s).split("-")[0][4:] if str(s).startswith("tess") else "" for s in ffi_file]
        )

    return {
        "t": time,
        "spacecrafttime": time - timecorr,
        "timecorr": timecorr,
        "quality": quality,
        "cadenceno": cadenceno,
        "fcirc": fcirc,
        "fpsf": fpsf,
        "fcircr": fcircr,
        "fpsfr": fpsfr,
        "err_circ_photon": err_circ,
        "err_psf_photon": err_psf,
        "err_circ_empirical": emp_circ,
        "err_psf_empirical": emp_psf,
        "medians": medians,
        "robmeans": robmeans,
        "xcms": xcms,
        "ycms": ycms,
        "circmasks": circmasks,
        "psfmasks": psfmasks,
        "timage": np.nan_to_num(timage),
        "sector": sector,
        "camera": camera,
        "ccd": ccd,
        "ffidatetime": ffidatetime,
        "xcenter": xcenter,
        "ycenter": ycenter,
        "xm": xm,
        "ym": ym,
        "prf_fit": pfit,
        "header": header,
        "ext1header": ext1header,
        "apheader": apheader,
        "n_bkg_pixels": n_bkg,
    }
