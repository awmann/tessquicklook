# TOI-2076 e validation

Scripts used to sanity-check the tessquicklook (Python) pipeline against two
independent references for TIC 27491137 / TOI-2076. Generated plots and large
intermediate data live under `output/toi2076_e_validation/` (git-ignored,
reproducible by rerunning these scripts) and `Validation/` (git-ignored,
reproducible via `tests/run_batch.py`).

Run `fetch_data.py` first to download the raw SAP/PDCSAP light curves these
scripts need (writes into `output/toi2076_e_validation/`); the IDL reference
in `IDLversions/27491137_idl.csv` and the tessquicklook output in
`Validation/TIC27491137.csv` are read directly.

## Paper claim check (arXiv:2505.06358)

The paper claims TOI-2076 e (P=3.0223445 d, T0=1740.21306 BTJD, dur=0.0902 d)
is recoverable only in a custom Vanderburg-style pipeline, not in SAP/PDCSAP.

- `dip_check.py` -- quantifies how much PDCSAP flattens the star's real
  rotational dips relative to SAP and the tessquicklook output. **Confirmed**:
  up to 61% depth suppression in PDCSAP, supporting the paper's stated
  mechanism (PDC over-subtracts real signal on this active young star).
- `make_fig1.py` -- reproduces the paper's Figure 1 style comparison
  (raw / detrended / phase-folded) for Sector 23.
- `make_full.py` -- same phase-fold comparison across all 4 sectors
  (16, 23, 50, 77).
- `calibrate.py` -- calibrates the phase-fold depth statistic against a null
  distribution of random trial periods (naive significance is not trustworthy
  on TESS red noise without this).
- `bls_search.py` / `bls_masked.py` -- blind BLS periodogram search for the
  3.02 d signal, the second masking the three known larger planets (b/c/d).
  **Inconclusive**: from-scratch detrending + BLS hits red-noise/aliasing
  artifacts in all three pipelines on this sparse, multi-year-baseline
  dataset, so the specific "SNR~10, custom-only" claim was not confirmed or
  refuted. A faithful test would need the paper's actual `notch` + BLS
  method (see `notch_locor_install_recipe` in project memory).

## IDL vs. Python cross-check

`idl_vs_python.py` / `idl_vs_python_binned.py` -- compares the Python port's
output against `27491137_idl.csv` (the original IDL pipeline's output for the
same target) per sector. **Result**: essentially identical where cadence
matches (correlation 1.000, sectors 16/23 at native 120s); correlation
0.9996-0.9999 for sectors 50/77 once the Python 20s output is binned to the
IDL's native 60s. Confirms the port reproduces the IDL pipeline's science
output.
