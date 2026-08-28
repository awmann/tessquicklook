"""Minimal working example: one target, one sector, from scratch.

    python examples/minimal_example.py

Produces `example_lightcurve.png` and `example_lightcurve.csv` next to this
file.  See `expected_output.png` in this directory for what it should look
like.

``quicklooktess`` picks the fastest data each sector offers.  TIC 88785435 was
never on a SPOC target list, so there is nothing faster than the FFIs here and
the call falls back to FFI photometry -- which is what makes it a good first
example: it exercises the whole photometry path.  For a target *with* SPOC
coverage, see `cadence_example.py`.

First run downloads, and caches under ~/.tessquicklook (see README):
  * a TESScut cutout                 ~2 MB
  * the TESS PRF grid for one CCD    ~6 MB
  * the sector's quaternion file     ~430 MB   <-- the slow part, once per sector
Afterwards the same call takes a few seconds.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Works whether or not the package has been pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tessquicklook import quicklooktess  # noqa: E402

TIC = 88785435   # a young, spotted star -- clear 1.8% rotational variability
SECTOR = 11      # 30-minute FFI cadence

result = quicklooktess(
    TIC,
    only_sectors=[SECTOR],
    corrndays=0.3,     # flexibility of the variability model, in days
    usecbv=True,
    ffi_options=dict(xsize=15, ysize=15, skew=True, kurt=True),
    outfile="example_lightcurve.csv",
)

t = result["t"]
flux = result["fcor"]          # systematics removed, variability retained
sector = result["sectors"][0]

p2p = np.median(np.abs(np.diff(flux))) * 1.48 / np.sqrt(2)
print(f"\n{len(t)} cadences over {np.ptp(t):.1f} days")
print(f"aperture chosen: {'circular' if sector['usecirc'] else 'PRF'} #{sector['best']}")
print(f"point-to-point scatter: {p2p * 1e6:.0f} ppm")
print(f"variability retained:   {np.std(flux) * 1e6:.0f} ppm rms")

fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
ax.scatter(t, flux, s=5, color="#eb6834", alpha=0.6, linewidths=0)
ax.set_xlabel("Time (BTJD)")
ax.set_ylabel("Normalised flux")
ax.set_title(f"TIC {TIC} — Sector {SECTOR}", loc="left")
fig.savefig("example_lightcurve.png", dpi=150)
print("\nwrote example_lightcurve.png and example_lightcurve.csv")
