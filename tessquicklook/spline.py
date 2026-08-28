"""B-spline machinery: ``keplerspline`` and the ``calcafull`` design matrix.

This is the piece that makes the pipeline's headline property work.  The IDL
``keplerspline(..., /calcafull, afullout=afull)`` call returns the *design
matrix* of the B-spline basis rather than a fitted curve, and
``decorrelatehr858`` column-stacks that matrix with the quaternion/CBV vectors
so the stellar variability and the instrumental systematics are solved for in a
single linear least-squares problem.  Fitting them jointly is what stops the
variability model from absorbing systematics (and vice versa), which is the
bias that afflicts PDCSAP for young, rapidly-rotating stars.

Fidelity note
-------------
``afull`` is used purely as a linear basis.  The joint solution's split between
"spline part" and "systematics part" depends only on the *column space* of that
block, not on the particular parameterisation, so reproducing IDL's exact
banded-matrix column ordering is unnecessary -- reproducing its knot vector and
spline order is both necessary and sufficient.  ``bspline_breakpoints`` below
therefore replicates ``bspline_bkpts.pro`` exactly, including its
single-precision breakpoint cast.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline

from .idlcompat import robust_mean

__all__ = [
    "bspline_breakpoints",
    "bspline_design_matrix",
    "keplerspline",
    "keplerspline_design",
]


def bspline_breakpoints(x, bkspace, nord=4, bkspread=1.0):
    """Port of ``bspline_bkpts.pro`` for the ``bkspace`` branch.

    Returns the padded ``fullbkpt`` knot vector.  Reproduces two IDL details
    that shift knot positions slightly:

    * ``nbkpts = long(range/float(bkspace)) + 1`` truncates toward zero.
    * ``bkpt = float(bkpt)`` casts breakpoints to single precision.
    """
    x = np.asarray(x, dtype=float)
    rng = x.max() - x.min()
    startx = x.min()

    nbkpts = int(rng / np.float32(bkspace)) + 1
    if nbkpts < 2:
        nbkpts = 2
    tempbkspace = float(rng) / float(nbkpts - 1)
    bkpt = np.arange(nbkpts, dtype=float) * tempbkspace + startx

    # IDL: bkpt = float(bkpt)
    bkpt = bkpt.astype(np.float32).astype(np.float64)

    # Guarantee coverage of the data range.
    if x.min() < bkpt.min():
        bkpt[np.argmin(bkpt)] = x.min()
    if x.max() > bkpt.max():
        bkpt[np.argmax(bkpt)] = x.max()

    # IDL does the padding arithmetic in single precision too, because `bkpt`
    # has already been cast to float at this point.  Matching that keeps the
    # knot vector agreement at the 1e-8 level instead of 1e-7.
    b32 = bkpt.astype(np.float32)
    nshortbkpt = b32.size
    step = np.float32(bkspread) if nshortbkpt == 1 else np.float32(
        (b32[1] - b32[0]) * np.float32(bkspread)
    )

    left = [np.float32(b32[0] - step * np.float32(i)) for i in range(nord - 1, 0, -1)]
    right = [np.float32(b32[-1] + step * np.float32(i)) for i in range(1, nord)]
    return np.concatenate([left, b32, right]).astype(np.float64)


def bspline_design_matrix(x, fullbkpt, nord=4):
    """Dense B-spline design matrix, shape ``(len(x), nbasis)``.

    ``nbasis = len(fullbkpt) - nord``, matching IDL's ``ncols(afull)``.
    """
    x = np.asarray(x, dtype=float)
    k = nord - 1
    # Guard against x sitting exactly on the last usable knot.
    return np.asarray(
        BSpline.design_matrix(np.clip(x, fullbkpt[k], fullbkpt[-k - 1]), fullbkpt, k).todense()
    )


def _split_at_gaps(t, bpndays, breakp=None):
    """Replicate keplerspline's segmentation.

    ``gaps = where(diff(t) gt bpndays)`` plus any caller-supplied break points;
    the returned list holds ``(start, stop)`` index pairs (stop exclusive).
    """
    t = np.asarray(t, dtype=float)
    difft = np.diff(t)
    gaps = list(np.flatnonzero(difft > bpndays))
    if breakp is not None and len(breakp):
        gaps = sorted(set(gaps) | set(int(b) for b in breakp))
    edges = [0] + [int(g) + 1 for g in gaps] + [t.size]
    # IDL indexes gaps[i-1]:gaps[i]-1; its `gaps` entries are the index *before*
    # the jump, and the slice runs to that index inclusive, so segment k spans
    # [edges[k], edges[k+1]).
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1) if edges[i + 1] > edges[i]]


def _keplerspline2(t, f, ndays, maxiter, include, sigmacut):
    """Port of ``keplerspline2`` -- iterated spline fit on one contiguous chunk.

    Returns ``(model, good_index)``.  Time is normalised to [0, 1] and the
    breakpoint spacing scaled accordingly, exactly as the IDL does.
    """
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    n = t.size
    span = t.max() - t.min()
    if span <= 0 or n < 4:
        return np.full(n, np.nanmedian(f) if n else np.nan), np.arange(n)

    t2 = (t - t.min()) / span
    bksp = ndays / span

    fullbkpt = bspline_breakpoints(t2, bksp, nord=4)
    basis_all = bspline_design_matrix(t2, fullbkpt, nord=4)

    lastgood = np.arange(n) if include is None else np.asarray(include, dtype=int)
    good = lastgood
    model = np.zeros(n)

    for _ in range(int(maxiter) + 1):
        if lastgood.size < 4:
            break
        A = basis_all[lastgood]
        y = f[lastgood]
        # Drop basis functions with no support in the current good set, so the
        # normal equations stay non-singular (IDL drops low-influence bkpts).
        used = np.flatnonzero(np.abs(A).sum(axis=0) > 0)
        coef, *_ = np.linalg.lstsq(A[:, used], y, rcond=None)
        model = basis_all[:, used] @ coef

        _, _, good = robust_mean(f - model, sigmacut)
        if include is not None:
            good = np.intersect1d(good, include)
        if good.size == lastgood.size and np.array_equal(good, lastgood):
            break
        if good.size < 4:
            good = lastgood
            break
        lastgood = good

    return model, good


def keplerspline(
    t,
    f,
    ndays=1.5,
    maxiter=5,
    sigmacut=3.0,
    bpndays=None,
    breakp=None,
    include=None,
):
    """Port of ``keplerspline.pro``.

    Fits an iteratively outlier-clipped cubic B-spline, independently within
    each contiguous chunk of data (chunks are split wherever the time gap
    exceeds ``bpndays``).  Returns ``(model, good_index, rms)``.
    """
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    if bpndays is None:
        bpndays = ndays

    order = np.argsort(t, kind="stable")
    resort = not np.all(np.diff(t) >= 0)
    if resort:
        t, f = t[order], f[order]
        if include is not None:
            flag = np.zeros(t.size, dtype=bool)
            flag[include] = True
            include = np.flatnonzero(flag[order])

    model = np.zeros(t.size)
    goodind = []
    for start, stop in _split_at_gaps(t, bpndays, breakp):
        seg_inc = None
        if include is not None:
            sel = include[(include >= start) & (include < stop)] - start
            seg_inc = sel
        m, g = _keplerspline2(
            t[start:stop], f[start:stop], ndays, maxiter, seg_inc, sigmacut
        )
        model[start:stop] = m
        goodind.append(g + start)

    goodind = np.concatenate(goodind) if goodind else np.array([], dtype=int)
    rms = np.std(model[goodind] - f[goodind], ddof=1) if goodind.size > 1 else np.nan

    if resort:
        inverse = np.argsort(order, kind="stable")
        model = model[inverse]
        flag = np.zeros(t.size, dtype=bool)
        flag[goodind] = True
        goodind = np.flatnonzero(flag[inverse])

    return model, goodind, rms


def keplerspline_design(t, ndays=1.5, bpndays=None, breakp=None):
    """The ``/calcafull`` path: return the spline design matrix, not a fit.

    Mirrors ``keplerspline(..., /calcafull, afullout=afull)``, which forces
    ``maxiter=0`` and includes every point so the basis covers outliers too.

    Because each data chunk is fit with its own independent spline, the
    assembled matrix is **block diagonal** -- chunk *k* occupies its own set of
    columns.  That block structure is what keeps the variability model local
    across download gaps, so it is reproduced here.

    Returns an array of shape ``(len(t), total_basis_functions)``.
    """
    t = np.asarray(t, dtype=float)
    if bpndays is None:
        bpndays = ndays

    order = np.argsort(t, kind="stable")
    resort = not np.all(np.diff(t) >= 0)
    ts = t[order] if resort else t

    blocks, rows = [], []
    for start, stop in _split_at_gaps(ts, bpndays, breakp):
        seg = ts[start:stop]
        span = seg.max() - seg.min() if seg.size else 0.0
        if seg.size < 4 or span <= 0:
            # Degenerate chunk: a single constant column (IDL stores a scalar 1).
            blocks.append(np.ones((seg.size, 1)))
        else:
            t2 = (seg - seg.min()) / span
            fullbkpt = bspline_breakpoints(t2, ndays / span, nord=4)
            blocks.append(bspline_design_matrix(t2, fullbkpt, nord=4))
        rows.append((start, stop))

    ncols = sum(b.shape[1] for b in blocks)
    afull = np.zeros((ts.size, ncols))
    col = 0
    for (start, stop), b in zip(rows, blocks):
        afull[start:stop, col : col + b.shape[1]] = b
        col += b.shape[1]

    if resort:
        afull = afull[np.argsort(order, kind="stable")]
    return afull
