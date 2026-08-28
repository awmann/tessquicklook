"""Compare the Python spline port against reference output from real IDL.

Run idl_compare/test_spline.pro first to regenerate the reference files.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import subspace_angles

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tessquicklook.spline import (  # noqa: E402
    bspline_breakpoints,
    bspline_design_matrix,
    keplerspline,
)

CMP = Path(__file__).resolve().parents[1] / "idl_compare"


def _require_reference(*names):
    """Exit cleanly when the IDL reference data is absent.

    ``idl_compare/`` is not redistributable and so is not part of the public
    repository (see the README).  A fresh clone therefore cannot run this
    check -- that is expected, and must not look like a test failure.
    """
    missing = [n for n in names if not (CMP / n).exists()]
    if missing:
        print(f"SKIPPED: {__doc__.splitlines()[0]}")
        print(f"  needs IDL reference data not distributed with this repo: "
              f"{', '.join(missing)}")
        print(f"  expected under {CMP}/ -- regenerate with idl_compare/run_idl.sh")
        print("  (requires IDL plus the original library; see the README)")
        raise SystemExit(0)

def main():
    _require_reference('spline_input.txt', 'spline_output.txt', 'spline_meta.txt', 'afull_matrix.txt', 'afull_bkpt.txt', 'afull_shape.txt', 'afull_proj.txt')

    t, f = np.loadtxt(CMP / "spline_input.txt", unpack=True)
    s1_idl, s2_idl = np.loadtxt(CMP / "spline_output.txt", unpack=True)

    ok = True

    # ---- keplerspline at two smoothing scales -----------------------------
    for ndays, ref, label in ((0.5, s1_idl, "ndays=0.5"), (1.5, s2_idl, "ndays=1.5")):
        model, good, rms = keplerspline(t, f, ndays=ndays)
        resid = model - ref
        # Scale relative to the variability amplitude actually being modelled.
        rel = np.max(np.abs(resid)) / np.ptp(ref)
        print(f"keplerspline {label}: max|Δ| = {np.max(np.abs(resid)):.3e}  "
              f"({rel * 100:.4f}% of model range), rms Δ = {np.std(resid):.3e}")
        if rel > 1e-3:
            ok = False
            print(f"   ^^ EXCEEDS tolerance")

    # ---- breakpoint vector -------------------------------------------------
    n1 = 400
    tt = t[:n1]
    span = tt.max() - tt.min()
    t2 = (tt - tt.min()) / span
    bkpt_idl = np.loadtxt(CMP / "afull_bkpt.txt")
    bkpt_py = bspline_breakpoints(t2, 0.3 / span, nord=4)
    print(f"\nbreakpoints: IDL n={bkpt_idl.size}, Python n={bkpt_py.size}, "
          f"max|Δ| = {np.max(np.abs(bkpt_idl - bkpt_py)):.3e}"
          if bkpt_idl.size == bkpt_py.size else
          f"\nbreakpoints: SIZE MISMATCH IDL {bkpt_idl.size} vs Python {bkpt_py.size}")
    if bkpt_idl.size != bkpt_py.size or np.max(np.abs(bkpt_idl - bkpt_py)) > 1e-6:
        ok = False

    # ---- design matrix shape ----------------------------------------------
    ncols_idl, nrows_idl, nbkpt_idl = np.loadtxt(CMP / "afull_shape.txt")
    A = bspline_design_matrix(t2, bkpt_py, nord=4)
    print(f"afull shape: IDL ({int(ncols_idl)} basis x {int(nrows_idl)} pts), "
          f"Python {A.shape[1]} basis x {A.shape[0]} pts")
    if A.shape != (int(nrows_idl), int(ncols_idl)):
        ok = False
        print("   ^^ SHAPE MISMATCH")

    # ---- column-space equivalence (the property that actually matters) -----
    # The joint fit only cares about afull's column space, so the correct
    # basis-independent test is the principal angle between the subspaces.
    # IDL accumulates afull in single precision (fltarr), so agreement is
    # bounded by float32 epsilon (~1e-7), not float64.
    B = np.loadtxt(CMP / "afull_matrix.txt")
    ang = subspace_angles(A, B).max()
    print(f"max principal angle between column spaces: {ang:.3e} rad "
          f"(float32 epsilon ~1.2e-7)")
    if ang > 1e-6:
        ok = False
        print("   ^^ EXCEEDS tolerance")

    print(f"partition of unity: Python row-sum err {np.max(np.abs(A.sum(1) - 1)):.2e}, "
          f"IDL {np.max(np.abs(B.sum(1) - 1)):.2e}")

    proj_idl = np.loadtxt(CMP / "afull_proj.txt")
    y = np.sin(2 * np.pi * (tt - 1600.0) / 2.3)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    d = np.max(np.abs(A @ coef - proj_idl))
    print(f"column-space projection: max|Δ| = {d:.3e} of a unit-amplitude signal "
          f"(TESS FFI scatter is ~1e-3)")
    if d > 1e-5:
        ok = False
        print("   ^^ EXCEEDS tolerance")

    print("\n" + ("ALL SPLINE CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
