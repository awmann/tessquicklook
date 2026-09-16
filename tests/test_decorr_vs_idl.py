"""Compare the Python decorrelation port against reference output from real IDL."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tessquicklook.decorrelate import decorrelate, quatcorrect_one  # noqa: E402

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
    _require_reference('decorr_input.txt', 'decorr_output.txt', 'decorr_coeffs.txt', 'decorr_meta.txt')

    data = np.loadtxt(CMP / "decorr_input.txt")
    t, f, vecs = data[:, 0], data[:, 1], data[:, 2:]
    model_idl, sysmodel_idl, corr_idl = np.loadtxt(CMP / "decorr_output.txt", unpack=True)
    coeffs_idl = np.loadtxt(CMP / "decorr_coeffs.txt")
    ncols_idl, nrows_idl, ncoef_idl = np.loadtxt(CMP / "decorr_meta.txt")

    # quatcorrectonelc mean-subtracts each quaternion vector before fitting.
    mvecs = vecs - vecs.mean(axis=0)

    coeffs, design, ncol_var, good = decorrelate(
        t, f - 1.0, mvecs, order=2, torder=5, maxiter=5
    )

    ok = True
    print(f"design matrix: IDL {int(ncols_idl)} cols x {int(nrows_idl)} rows, "
          f"Python {design.shape[1]} x {design.shape[0]}")
    if design.shape != (int(nrows_idl), int(ncols_idl)):
        ok = False
        print("   ^^ SHAPE MISMATCH")
    print(f"variability block = {ncol_var} cols (const + torder=5)")

    dc = np.max(np.abs(coeffs - coeffs_idl))
    scale = np.max(np.abs(coeffs_idl))
    print(f"\ncoefficients: max|Δ| = {dc:.3e}  (largest |coeff| = {scale:.3e}, "
          f"relative {dc / scale:.2e})")
    if dc / scale > 1e-8:
        ok = False
        print("   ^^ EXCEEDS tolerance")

    model = design @ coeffs
    sysmodel = design[:, ncol_var:] @ coeffs[ncol_var:]
    corrected = f - sysmodel

    for label, py, ref in (
        ("full model", model, model_idl),
        ("systematics model", sysmodel, sysmodel_idl),
        ("corrected flux", corrected, corr_idl),
    ):
        d = np.max(np.abs(py - ref))
        print(f"{label:20s}: max|Δ| = {d:.3e}")
        if d > 1e-10:
            ok = False
            print("   ^^ EXCEEDS tolerance")

    # The high-level wrapper must reproduce the same thing.
    c2, s2, m2, _, _ = quatcorrect_one(t, f, mvecs, order=2, torder=5)
    d = np.max(np.abs(c2 - corr_idl))
    print(f"{'quatcorrect_one':20s}: max|Δ| = {d:.3e}")
    if d > 1e-10:
        ok = False

    print(f"\nclipping: {good.size}/{t.size} points retained after 3-sigma iteration")
    print("\n" + ("ALL DECORRELATION CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
