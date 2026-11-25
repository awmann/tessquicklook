"""Per-cadence scattered-light flags.

Why: TESScut FFI cutouts carry only SPOC's camera-level, *predicted* Straylight
bit (2048), mostly on cameras 1/2, and never the Straylight2 bit (4096) that
SPOC 2-min data puts on the real sky ramps.  The FFI quality mask keeps
``quality == 0`` (as the IDL does), so scattered-light ramps reach the light
curve unflagged.  The 2-min mask keeps 32768 cadences, which also carry
scattered light.

These flags are *output only*: they never change which cadences are kept or
which enter the systematics fit.  (Excluding flagged cadences from the fit was
tested on 100 TIDYE curves: 1-h CDPP unchanged, flagged cadences no cleaner.)

Sky excess
    e(t) = sky(t) / b - 1,  b = 10th percentile of sky over quality == 0
    FFI   : sky = 20th-percentile pixel of the stamp, real (non-padded) pixels only
    2-min : sky = SPOC SAP_BKG
    e = 1 means the sky has doubled.  NaN when no usable sky exists (stamp off
    the science pixels, baseline <= MIN_BASELINE, or too few clean cadences).

Bits of ``sl_flag`` (uint8)
    1   e > DEFAULT_X (1.0); FFI dilated by one cadence each side
    8   e > LOOSE_X (0.3); same dilation
    2   SPOC QUALITY has 32768 (2-min only; kept by the 2-min mask)
    4   SPOC QUALITY has 2048/4096/8192 (2-min only; normally empty because the
        mask drops them, but set if allowscatteredlight keeps them)
    128 no usable sky for this sector (bits 1/8 cannot be evaluated)

Calibration against SPOC 2-min 2048|4096 flags (400 light curves, 273 CCDs,
2026-11-18), recall / fraction of SPOC-clean cadences flagged:
    X     FFI single stamp       2-min SAP_BKG
    0.3   0.996 / 0.125          0.999 / 0.132
    1.0   0.914 / 0.032          0.983 / 0.029
FFI numbers are for stamps with a baseline above MIN_BASELINE (1006 of 1074);
the other 6% lay off the science pixels (whole stamp ~0 e-/s) and get bit 128.
The 3-stamp CCD median used for sl_mask 0.1.0 scored 0.917 / 0.031 at X=1.
"""
import numpy as np

VERSION = "sl_flags 1 (tessquicklook 0.3.0): bit1 e>1.0, bit8 e>0.3, bit2 SPOC 32768, bit4 SPOC 2048|4096|8192, bit128 no sky"
DEFAULT_X = 1.0
LOOSE_X = 0.3
FFI_DILATE = 1           # cadences each side, FFI only
MIN_BASELINE = 1.0       # e-/s; below this the stamp has no sky (off the science pixels)
MIN_GOOD = 50            # clean cadences needed to set the baseline

BIT_SKY = 1
BIT_SPOC_32768 = 2
BIT_SPOC_STRAY = 4
BIT_SKY_LOOSE = 8
BIT_NO_SKY = 128


def stamp_sky(flux_cube):
    """20th-percentile pixel per cadence over the stamp's real pixels.

    Edge-of-detector stamps are zero- or NaN-padded; those pixels have a
    time-median of exactly 0 or NaN and are excluded.  Returns NaNs if fewer
    than 20 real pixels remain.
    """
    flux_cube = np.asarray(flux_cube, dtype=float)
    flat = flux_cube.reshape(flux_cube.shape[0], -1)
    with np.errstate(all="ignore"):
        pm = np.nanmedian(flat, axis=0)
    real = np.isfinite(pm) & (pm != 0)
    if real.sum() < 20:
        return np.full(flux_cube.shape[0], np.nan)
    with np.errstate(all="ignore"):
        return np.nanpercentile(flat[:, real], 20, axis=1)


def sky_excess(sky, quality):
    """(excess, baseline).  excess is all-NaN when the baseline is unusable."""
    sky = np.asarray(sky, dtype=float)
    good = (np.asarray(quality) == 0) & np.isfinite(sky)
    if good.sum() < MIN_GOOD:
        return np.full(sky.size, np.nan), np.nan
    b = float(np.nanpercentile(sky[good], 10))
    if not np.isfinite(b) or b <= MIN_BASELINE:
        return np.full(sky.size, np.nan), b
    return sky / b - 1.0, b


def _dilate(b, n):
    if n <= 0 or not b.any():
        return b
    out = b.copy()
    for k in range(1, n + 1):
        out[k:] |= b[:-k]
        out[:-k] |= b[k:]
    return out


def excess_bits(excess, dilate=0):
    """Bits 1 and 8 from an excess array on the full cadence grid (time-ordered)."""
    e = np.asarray(excess, dtype=float)
    if not np.isfinite(e).any():
        return np.full(e.size, BIT_NO_SKY, dtype=np.uint8)
    e0 = np.nan_to_num(e, nan=0.0)
    out = np.where(_dilate(e0 > DEFAULT_X, dilate), BIT_SKY, 0)
    out |= np.where(_dilate(e0 > LOOSE_X, dilate), BIT_SKY_LOOSE, 0)
    return out.astype(np.uint8)


def spoc_bits(quality):
    q = np.asarray(quality, dtype=np.int64)
    out = np.where((q & 32768) > 0, BIT_SPOC_32768, 0)
    out |= np.where((q & (2048 | 4096 | 8192)) > 0, BIT_SPOC_STRAY, 0)
    return out.astype(np.uint8)


def ffi_flags(flux_cube, quality, time=None):
    """(sky, excess, sl_flag) for an FFI cutout on its full cadence grid."""
    sky = stamp_sky(flux_cube)
    exc, _ = sky_excess(sky, quality)
    if time is not None:
        o = np.argsort(np.nan_to_num(np.asarray(time, float), nan=np.inf), kind="stable")
        flag = np.empty(exc.size, np.uint8)
        flag[o] = excess_bits(exc[o], FFI_DILATE)
    else:
        flag = excess_bits(exc, FFI_DILATE)
    return sky, exc.astype(np.float32), flag


def spoc_flags(sap_bkg, quality):
    """(excess, sl_flag) for a SPOC 2-min/20-s light curve on its full grid."""
    exc, _ = sky_excess(sap_bkg, quality)
    flag = excess_bits(exc, 0) | spoc_bits(quality)
    return exc.astype(np.float32), flag.astype(np.uint8)
