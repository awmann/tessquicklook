"""Choosing a cadence: the same star at 20 s, 120 s and FFI.

    python examples/cadence_example.py

HIP 67522 (TIC 166527623) has SPOC 120 s light curves in Sectors 11 and 38 and
20 s light curves in 64, 101 and 102 -- so ``cadence="auto"`` produces a mixed
light curve, and the ``cadence_s`` column records which is which.

This script plans the whole target, then runs one sector three ways so the
trade-off is visible: 20 s data resolves flare morphology and ingress shape that
200 s FFI photometry averages over, at the cost of higher per-point scatter.

Writes `cadence_example.png` next to this file.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tessquicklook import plan_cadences, quicklooktess  # noqa: E402
from tessquicklook.idlcompat import point_to_point_scatter as p2p  # noqa: E402

TIC = 166527623
SECTOR = 64

# What would an "auto" run over the whole target do?  No data is downloaded.
plan = plan_cadences(TIC)
print("cadence plan for all sectors:")
for source, sectors in plan.items():
    label = "FFI" if source == "ffi" else f"{source:.0f} s SPOC"
    print(f"  {label:>12}: {sectors}")

# Now one sector, three ways.
runs = {}
for label, cadence in (("20 s", "fast"), ("120 s", "short"), ("FFI 200 s", "ffi")):
    print(f"\n=== {label} ===")
    runs[label] = quicklooktess(
        TIC, cadence=cadence, only_sectors=[SECTOR],
        ffi_options=dict(xsize=15, ysize=15, skew=True, kurt=True),
    )

print(f"\n{'source':>10} {'points':>8} {'p2p native':>13} {'-> per 30 min':>15} "
      f"{'30-min bin p2p':>16}")
for label, r in runs.items():
    native = p2p(r["fcor"])
    cadence_s = np.median(r["cadence_s"])
    # Scale the per-point scatter to a common 30-minute integration.  This is
    # the fair noise comparison between cadences.
    per30 = native * np.sqrt(cadence_s / 1800.0)

    # Binning to 30 min and taking the point-to-point of *that* measures
    # something else entirely -- the star moves ~1% between adjacent 30-minute
    # bins, so this number is stellar rotation, not noise.  Included precisely
    # because all three agree on it: same star, same signal.
    edges = np.arange(r["t"].min(), r["t"].max(), 30.0 / 60 / 24)
    which = np.digitize(r["t"], edges)
    binned = np.array([r["fcor"][which == i].mean()
                       for i in range(1, edges.size) if np.any(which == i)])
    print(f"{label:>10} {r['t'].size:>8} {native * 1e6:9.0f} ppm "
          f"{per30 * 1e6:11.0f} ppm {p2p(binned) * 1e6:12.0f} ppm")

fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
colours = {"20 s": "#2a78d6", "120 s": "#eb6834", "FFI 200 s": "#0b0b0b"}
t0 = min(r["t"].min() for r in runs.values())
for label, r in runs.items():
    m = r["t"] < t0 + 1.5
    ax.scatter(r["t"][m], r["fcor"][m], s=4, alpha=0.5, linewidths=0,
               color=colours[label], label=label)
ax.set_xlabel("Time (BTJD)")
ax.set_ylabel("Normalised flux")
ax.set_title(f"HIP 67522, Sector {SECTOR} — same correction, three cadences",
             loc="left")
ax.legend(frameon=False, markerscale=4)
fig.savefig(Path(__file__).parent / "cadence_example.png", dpi=150)
print("\nwrote cadence_example.png")
