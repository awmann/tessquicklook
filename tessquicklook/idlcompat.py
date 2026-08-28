"""Faithful Python equivalents of the IDL primitives used by the pipeline.

Every function here mirrors a specific IDL routine.  Where IDL semantics are
subtle (single-precision breakpoints, two-pass sigma clipping, ``interpolate``'s
cubic convolution) the behaviour is reproduced exactly rather than replaced with
the nearest NumPy idiom, because the pipeline's outputs depend on it.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "logspace_idl",
    "robust_mean",
    "point_to_point_scatter",
    "cdpp",
    "fillarr",
    "contiguous_region",
    "cm_nan",
    "median_even",
]


def median_even(a):
    """IDL ``MEDIAN(/EVEN)`` -- averages the two middle values for even N.

    Plain ``MEDIAN`` in IDL returns the upper of the two middle elements, but
    ``robust_mean`` uses ``/EVEN``, which matches ``numpy.median``.
    """
    return np.median(a)


def median_idl(a, axis=None):
    """IDL ``MEDIAN()`` without /EVEN.

    For an even number of elements IDL returns element ``N/2`` of the sorted
    array (the upper middle value), *not* the average of the two middle values.
    """
    a = np.asarray(a)
    if axis is None:
        flat = np.sort(a, axis=None)
        n = flat.size
        if n == 0:
            return np.nan
        return flat[n // 2]
    srt = np.sort(a, axis=axis)
    n = srt.shape[axis]
    return np.take(srt, n // 2, axis=axis)


def logspace_idl(first, last, n):
    """IDL ``logspace(first, last, n)`` from logspace.pro."""
    a = np.log10(first)
    b = np.log10(last)
    return 10.0 ** (np.arange(n, dtype=float) / (n - 1.0) * (b - a) + a)


def fillarr(step, start, stop):
    """IDL ``fillarr(step, start, stop)`` -- inclusive regular grid."""
    n = int(np.floor((stop - start) / step)) + 1
    return start + np.arange(n, dtype=float) * step


def robust_mean(y, cut):
    """Port of ``robust_mean.pro`` (Freudenreich/Landsman).

    Returns ``(mean, sigma, good_index)``.  The routine performs *two* passes:
    an initial cut on the median absolute deviation, then a second cut on the
    truncation-corrected standard deviation.  ``good_index`` is what the
    pipeline actually consumes, so both passes matter.
    """
    y = np.asarray(y, dtype=float)
    npts = y.size
    ymed = np.median(y)
    absdev = np.abs(y - ymed)
    medabsdev = np.median(absdev) / 0.6745
    if medabsdev < 1.0e-24:
        medabsdev = np.mean(absdev) / 0.8

    cutoff = cut * medabsdev
    good = np.flatnonzero(absdev <= cutoff)
    if good.size == 0:
        return np.nan, np.nan, good
    goodpts = y[good]
    mean = goodpts.mean()
    sigma = np.sqrt(np.sum((goodpts - mean) ** 2) / good.size)

    sc = max(cut, 1.0)
    if sc <= 4.50:
        sigma = sigma / (-0.15405 + 0.90723 * sc - 0.23584 * sc**2 + 0.020142 * sc**3)

    cutoff = cut * sigma
    good = np.flatnonzero(absdev <= cutoff)
    if good.size == 0:
        return np.nan, np.nan, good
    goodpts = y[good]
    mean = goodpts.mean()
    sigma = np.sqrt(np.sum((goodpts - mean) ** 2) / good.size)

    if sc <= 4.50:
        sigma = sigma / (-0.15405 + 0.90723 * sc - 0.23584 * sc**2 + 0.020142 * sc**3)
    sigma = sigma / np.sqrt(npts - 1.0)

    return mean, sigma, good


def point_to_point_scatter(f):
    """The scatter metric used by ``chooseaperturetess`` when /cdpp is off.

    ``median(|f[1:] - f[:-1]|) * 1.48 / sqrt(2)``.

    Note the IDL writes ``thisfc[0:n_elements(thisfc)-1]`` for the second term,
    which is the *whole* array rather than ``[0:-2]``.  IDL silently truncates
    the longer operand, so the effective operation is the intended successive
    difference; we implement the intent.
    """
    f = np.asarray(f, dtype=float)
    if f.size < 2:
        return np.inf
    return np.median(np.abs(f[1:] - f[:-1])) * 1.48 / np.sqrt(2.0)


def cdpp(lc, hours=6, cadence_days=None):
    """Port of ``cdpp.pro``, made cadence-aware.

    The IDL computes ``naverage = hours*2 + 1``, which hard-codes two samples
    per hour -- i.e. 30-minute cadence.  That was correct for the original FFIs
    but is wrong for every sector from 27 onward: at 10-minute cadence the
    window spans 130 minutes rather than 6 hours, and at 200 s only 43 minutes.
    Averaging over too short a window understates the correlated noise that
    large apertures suffer, which biases ``chooseaperturetess`` toward
    implausibly small apertures (a 1-pixel mask, in the sector-38 test case).

    Passing ``cadence_days`` scales the window to a true ``hours``-long
    average.  Omitting it reproduces the IDL's behaviour exactly.
    """
    lc = np.asarray(lc, dtype=float)
    if lc.size == 0:
        return np.nan
    if cadence_days is None:
        naverage = int(hours * 2 + 1)
    else:
        naverage = max(int(round(hours / 24.0 / float(cadence_days))), 1)
    n = lc.size - naverage
    if n <= 0:
        return np.nan
    # IDL stdev() is the sample standard deviation (N-1 denominator).
    cdppt = np.empty(n)
    for i in range(n):
        seg = lc[i : i + naverage + 1]
        cdppt[i] = 1e6 * np.std(seg, ddof=1)
    return np.median(cdppt) / np.sqrt(naverage)


def cm_nan(image):
    """NaN-tolerant centre of mass, IDL ``cm_nan``.

    Returns ``(xcm, ycm)`` in pixel coordinates using the same axis convention
    as ``extractphotometrytess`` (x = column index, y = row index).
    """
    img = np.asarray(image, dtype=float)
    good = np.isfinite(img)
    if not good.any():
        return np.nan, np.nan
    w = np.where(good, img, 0.0)
    total = w.sum()
    if total == 0:
        return np.nan, np.nan
    ny, nx = img.shape
    xs = np.arange(nx)[None, :]
    ys = np.arange(ny)[:, None]
    return float((w * xs).sum() / total), float((w * ys).sum() / total)


def contiguous_region(mask, xstart, ystart):
    """Port of ``contiguousregion.pro`` -- 4-connected flood fill.

    The IDL implementation recurses; we use an explicit stack so deep regions
    cannot blow the Python recursion limit.  ``mask`` is indexed ``[y, x]``.
    """
    mask = np.asarray(mask).astype(bool)
    out = np.zeros_like(mask, dtype=float)
    ny, nx = mask.shape
    xs, ys = int(round(xstart)), int(round(ystart))
    if not (0 <= xs < nx and 0 <= ys < ny):
        return out
    if not mask[ys, xs]:
        # IDL sets the seed pixel regardless of whether the map is set there.
        out[ys, xs] = 1.0
        return out

    stack = [(xs, ys)]
    while stack:
        x, y = stack.pop()
        if out[y, x]:
            continue
        out[y, x] = 1.0
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx_, ny_ = x + dx, y + dy
            if 0 <= nx_ < nx and 0 <= ny_ < ny and mask[ny_, nx_] and not out[ny_, nx_]:
                stack.append((nx_, ny_))
    return out
