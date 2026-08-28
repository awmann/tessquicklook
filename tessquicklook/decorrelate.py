"""The simultaneous systematics + variability fit.

Ports ``decorrelatehr858.pro``, ``quatcorrectonelc.pro`` and ``quatcorrect.pro``.

The pipeline's defining property lives here.  A single linear model

    flux - 1  ~  [ variability basis | systematics vectors ] @ coeffs

is solved by iteratively-clipped least squares, and then **only the systematics
part of the model is subtracted**.  Because the astrophysical variability is
free to move during the fit, the systematics coefficients are not biased by it
-- and, crucially, the variability is not distorted by having been flattened
first.  That is what separates this from PDCSAP for rapidly-rotating young
stars.

Variability basis
-----------------
``variability_basis`` selects what spans the astrophysical component:

``"poly"``
    Constant + ``torder`` powers of normalised, mean-centred time.  **This is
    what the IDL actually does in production.**  ``keplerspline(/calcafull)``
    is supposed to supply a B-spline basis instead, but ``bspline_iterfit``
    declares ``_EXTRA`` rather than ``_REF_EXTRA``, so the ``afullout`` output
    keyword never propagates back and ``afull`` is always undefined.
    ``decorrelatehr858`` therefore always takes its polynomial branch, and the
    ``corrndays`` argument has no effect on the correction.

``"spline"``
    The B-spline design matrix the code was written to use.  Far more flexible
    (``corrndays`` becomes meaningful), and appropriate for stars whose
    rotation a degree-5 polynomial cannot track over a sector.
"""

from __future__ import annotations

import numpy as np

from .idlcompat import robust_mean
from .spline import keplerspline_design

__all__ = ["build_systematics_vectors", "decorrelate", "quatcorrect_one", "quatcorrect"]

# Quaternion columns used by quatcorrectonelc, in IDL order.
_STD_KEYS = ("q1std", "q2std", "q3std", "q1q3std", "q1q2std", "q2q3std")
_MEAN_KEYS = ("q1mean", "q2mean", "q3mean", "q1q3mean", "q1q2mean", "q2q3mean")
_SKEW_KEYS = ("q1skew", "q2skew", "q3skew")
_KURT_KEYS = ("q1kurt", "q2kurt", "q3kurt")


def build_systematics_vectors(
    quats,
    rawdata=None,
    cbvs=None,
    bg=False,
    means=True,
    skew=False,
    kurt=False,
    center_cbvs=False,
):
    """Assemble the systematics design vectors, matching ``quatcorrectonelc``.

    Returns ``(vectors, names)`` where ``vectors`` has shape ``(npoints, nvec)``.

    Quaternion columns are mean-subtracted; CBVs are not (matching the FFI
    routine).  ``quicklooksector3.pro``, the short-cadence sibling, standardises
    every column including the CBVs, so ``center_cbvs=True`` reproduces that.
    The choice does not affect the science: the variability basis carries a
    constant term, so shifting a regressor only moves the systematics model by
    an offset, and the caller renormalises to a unit median afterwards.

    ``bg=True`` additionally includes the background median and robust mean,
    which is how the ``datamed`` variant of the light curve is produced.
    """
    cols, names = [], []

    def add(arr, name, center=True):
        a = np.asarray(arr, dtype=float)
        cols.append(a - a.mean() if center else a)
        names.append(name)

    for k in _STD_KEYS:
        add(quats[k], k)

    if bg:
        if rawdata is None:
            raise ValueError("bg=True requires rawdata for medians/robmeans")
        add(rawdata["medians"], "medians")
        add(rawdata["robmeans"], "robmeans")

    if means:
        for k in _MEAN_KEYS:
            add(quats[k], k)
    if skew:
        for k in _SKEW_KEYS:
            add(quats[k], k)
    if kurt:
        for k in _KURT_KEYS:
            add(quats[k], k)

    if cbvs is not None:
        for kk in range(1, 9):
            for prefix, label in (("v", "cbv"), ("b3v", "cbv_band3")):
                key = f"{prefix}{kk}"
                if key not in cbvs:
                    continue
                vec = np.asarray(cbvs[key], dtype=float)
                # IDL keeps a CBV only if it is fully finite and not all zero.
                if np.all(np.isfinite(vec)) and np.sum(np.abs(vec)) > 0:
                    add(vec, f"{label}{kk}", center=center_cbvs)

    if not cols:
        return np.zeros((len(quats["q1std"]), 0)), []
    return np.column_stack(cols), names


def _solve(X, y, method="normal", rcond=1e-10):
    """Least-squares solve with column equilibration.

    The design matrix is severely ill-conditioned in normal use.  The
    quaternion regressors differ in amplitude by ~11 orders of magnitude
    (``q1std`` ~5e-7 against ``q1q3std`` ~7e-12 against ``q1skew`` ~5e-1), and
    ``order=2`` squares that spread, driving ``cond(X'X)`` past 1e46 -- far
    beyond float64's ~1e16.  Solving the raw normal equations there yields
    coefficients of order 1e20 and a systematics model that is pure numerical
    noise, which *degrades* the light curve instead of cleaning it.

    Rescaling each column to unit norm leaves the fitted model and the
    variability/systematics split mathematically unchanged (the coefficients
    simply absorb the scale factors) while restoring numerical sanity.  A
    truncated SVD then discards genuinely degenerate directions rather than
    amplifying them.

    ``method="normal"`` still solves the (scaled) normal equations, matching
    IDL's ``invert()`` closely when the problem is well behaved; ``"lstsq"``
    forces the SVD path.
    """
    scale = np.sqrt((X**2).sum(axis=0))
    scale[scale == 0] = 1.0
    Xs = X / scale

    if method == "normal":
        ata = Xs.T @ Xs
        if np.linalg.cond(ata) < 1.0 / rcond:
            try:
                return np.linalg.solve(ata, Xs.T @ y) / scale
            except np.linalg.LinAlgError:
                pass

    coeffs, *_ = np.linalg.lstsq(Xs, y, rcond=rcond)
    return coeffs / scale


def decorrelate(
    time,
    flux,
    vectors,
    order=1,
    torder=2,
    maxiter=5,
    include=None,
    afull=None,
    sigmacut=3.0,
    solver="normal",
):
    """Port of ``decorrelatehr858.pro``.

    Returns ``(coeffs, design, ncol_variability, good)``.

    ``design`` columns are ordered ``[variability basis | vectors**1 | ...
    vectors**order]``, and ``ncol_variability`` marks the boundary -- callers
    need it to isolate the systematics half of the model.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim == 1:
        vectors = vectors[:, None]

    n = time.size
    span = time.max() - time.min()
    # IDL: ntime = (t-min)/span - (mean(t)-min)/span  -- normalised, mean-centred.
    ntime = (time - time.min()) / span - (time.mean() - time.min()) / span

    if afull is None:
        blocks = [np.ones((n, 1))]
        blocks += [(ntime**i)[:, None] for i in range(1, int(torder) + 1)]
        variability = np.hstack(blocks)
    else:
        # IDL replaces fullx entirely with afull; the B-spline basis is a
        # partition of unity, so it already spans the constant term.
        variability = np.asarray(afull, dtype=float)

    ncol_var = variability.shape[1]

    sysblocks = [vectors ** (i + 1) for i in range(int(order))]
    design = np.hstack([variability] + sysblocks) if sysblocks else variability

    lastgood = np.arange(n) if include is None else np.asarray(include, dtype=int)
    coeffs = None
    good = lastgood

    for _ in range(int(maxiter)):
        X = design[lastgood]
        y = flux[lastgood]
        coeffs = _solve(X, y, solver)
        model = design @ coeffs

        _, _, good = robust_mean(flux - model, sigmacut)
        if include is not None:
            good = np.intersect1d(good, include)
        if good.size == lastgood.size and np.array_equal(good, lastgood):
            break
        if good.size <= design.shape[1]:
            good = lastgood
            break
        lastgood = good

    return coeffs, design, ncol_var, good


def quatcorrect_one(
    time,
    flux,
    vectors,
    order=1,
    torder=2,
    afull=None,
    maxiter=5,
    include=None,
    solver="normal",
):
    """Port of ``quatcorrectonelc.pro``.

    Fits the joint model to ``flux - 1`` and returns
    ``(corrected_flux, sysmodel, fullmodel, coeffs)`` where
    ``corrected_flux = flux - sysmodel`` -- systematics removed, astrophysical
    variability retained.
    """
    flux = np.asarray(flux, dtype=float)
    coeffs, design, ncol_var, _ = decorrelate(
        time, flux - 1.0, vectors,
        order=order, torder=torder, maxiter=maxiter,
        include=include, afull=afull, solver=solver,
    )

    fullmodel = design @ coeffs
    # Systematics-only model: every column past the variability block.
    sysmodel = design[:, ncol_var:] @ coeffs[ncol_var:]
    return flux - sysmodel, sysmodel, fullmodel, coeffs


def quatcorrect(
    rawdata,
    quats,
    cbvs=None,
    order=2,
    torder=5,
    means=True,
    skew=False,
    kurt=False,
    bg=False,
    variability_basis="poly",
    corrndays=0.3,
    bpndays=None,
    maxiter=5,
    solver="normal",
    n_apertures=10,
):
    """Port of ``quatcorrect.pro`` -- correct all 20 aperture light curves.

    ``rawdata`` is the photometry dict from :mod:`tessquicklook.photometry`,
    holding ``t``, ``fcirc`` and ``fpsf`` arrays of shape ``(npoints, 10)``.

    Returns a dict with ``fcirccor``/``fpsfcor`` (systematics removed,
    renormalised to a median of 1) and ``fcirccorflat``/``fpsfcorflat``
    (residual about the full joint model).
    """
    t = np.asarray(rawdata["t"], dtype=float)
    n = t.size

    afull = None
    if variability_basis == "spline":
        afull = keplerspline_design(t, ndays=corrndays, bpndays=bpndays)
    elif variability_basis != "poly":
        raise ValueError("variability_basis must be 'poly' or 'spline'")

    vectors, vecnames = build_systematics_vectors(
        quats, rawdata=rawdata, cbvs=cbvs, bg=bg, means=means, skew=skew, kurt=kurt
    )

    out = {
        "t": t,
        "vector_names": vecnames,
        "variability_basis": variability_basis,
    }
    for kind in ("fcirc", "fpsf"):
        cor = np.full((n, n_apertures), np.nan)
        flat = np.full((n, n_apertures), np.nan)
        for i in range(n_apertures):
            f = np.asarray(rawdata[kind], dtype=float)[:, i]
            if not np.all(np.isfinite(f)):
                continue
            corrected, _, fullmodel, _ = quatcorrect_one(
                t, f, vectors, order=order, torder=torder,
                afull=afull, maxiter=maxiter, solver=solver,
            )
            cor[:, i] = corrected - np.median(corrected) + 1.0
            flat[:, i] = f - fullmodel
        out[kind] = np.asarray(rawdata[kind], dtype=float)
        out[kind + "cor"] = cor
        out[kind + "corflat"] = flat

    return out
