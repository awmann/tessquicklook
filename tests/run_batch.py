"""Batch-generate light curves for a list of TIC IDs, all available sectors.

    python tests/run_batch.py [TIC ...] [--cadence auto|fast|short|ffi|sc] [--out DIR]

With no TIC IDs it runs the built-in list.

Writes per-target CSV + PNG into output/lightcurves/ (or --out), plus a summary
table.  Resumable: targets whose CSV already exists are skipped.

The default is ``--cadence auto``, which takes the fastest data each sector
offers.  Products generated before v0.2.0 were FFI-only; pass ``--cadence ffi``
to reproduce those, and use ``--out`` to keep the two sets side by side rather
than mixing them in one directory.
"""

import argparse
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tessquicklook import quicklooktess  # noqa: E402
from tessquicklook.idlcompat import point_to_point_scatter as p2p  # noqa: E402

TICS = [410214986, 166527623, 250595513, 88785435, 27491137, 47319867,
        64837857, 286864983, 101011575, 146413471, 150151262, 434398831,
        143168991, 242295957, 242267549, 299798795, 257605131]

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "output" / "lightcurves"
OUT = DEFAULT_OUT

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880"
COR, GRID = "#eb6834", "#e5e4df"


def mad_std(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def plot(res, path):
    t, f = res["t"], res["fcor"]
    secs = sorted({s["sector"] for s in res["sectors"]})
    # One panel per contiguous group so long baselines stay legible.
    groups, cur = [], [0]
    for i in range(1, t.size):
        if t[i] - t[i - 1] > 20:
            groups.append((cur[0], i))
            cur = [i]
    groups.append((cur[0], t.size))

    n = len(groups)
    fig, axes = plt.subplots(n, 1, figsize=(11.5, 2.1 * n + 1.2),
                             layout="constrained", squeeze=False)
    axes = axes[:, 0]
    fig.patch.set_facecolor(SURFACE)
    for ax, (a, b) in zip(axes, groups):
        ax.scatter(t[a:b], f[a:b], s=3, color=COR, alpha=0.55, linewidths=0, zorder=3)
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=INK2, labelsize=8.5, length=3, width=0.8)
        ax.set_ylabel("norm. flux", color=INK2, fontsize=9)
    axes[-1].set_xlabel("Time (BTJD)", color=INK2, fontsize=10)
    star = res["star"]
    cadences = "/".join(f"{c:.0f}s" for c in np.unique(res["cadence_s"]))
    axes[0].set_title(
        f"TIC {res['ticid']}   Tmag {star['tmag']:.2f}   "
        f"{len(secs)} sectors: {secs}\n"
        f"{t.size} cadences ({cadences}) · point-to-point {p2p(f) * 1e6:.0f} ppm · "
        f"MAD scatter {mad_std(f) * 1e6:.0f} ppm",
        color=INK, fontsize=10.5, loc="left", pad=8)
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def main():
    global OUT

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tics", nargs="*", type=int,
                    help="TIC IDs to run; defaults to the built-in list")
    ap.add_argument("--cadence", default="auto",
                    help="auto (default), fast, short, sc, or ffi")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="output directory")
    args = ap.parse_args()

    tics = args.tics or TICS
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)

    log = OUT / "batch_log.txt"
    summary = []
    t_start = time.time()
    print(f"cadence={args.cadence}  ->  {OUT}\n")

    for i, tic in enumerate(tics, 1):
        csv = OUT / f"TIC{tic}.csv"
        png = OUT / f"TIC{tic}.png"
        tag = f"[{i}/{len(tics)}] TIC {tic}"

        if csv.exists() and png.exists():
            print(f"{tag}: already done, skipping", flush=True)
            continue

        t0 = time.time()
        try:
            res = quicklooktess(
                tic,
                cadence=args.cadence,
                usecbv=True,
                discard_quaternion_fits=True,   # keep the cache ~25 MB/sector
                ffi_options=dict(skew=True, kurt=True),
                outfile=str(csv),
                verbose=True,
            )
            plot(res, png)
            secs = sorted({s["sector"] for s in res["sectors"]})
            row = dict(tic=tic, tmag=res["star"]["tmag"], nsec=len(secs),
                       sectors=secs, n=res["t"].size,
                       p2p=p2p(res["fcor"]) * 1e6,
                       mad=mad_std(res["fcor"]) * 1e6,
                       mins=(time.time() - t0) / 60)
            summary.append(row)
            planned = {s for v in res.get("plan", {}).values() for s in v}
            lost = sorted(planned - set(secs))
            note = f", LOST {lost}" if lost else ""
            print(f"{tag}: OK  {len(secs)} sectors, {res['t'].size} cadences, "
                  f"p2p {row['p2p']:.0f} ppm, {row['mins']:.1f} min{note}", flush=True)
            # This module installs a blanket warnings filter, so a branch that
            # blew up would otherwise leave no trace at all.
            for f in res.get("failures", []):
                print(f"    branch failed: {f['source']} {f['sectors']} -- "
                      f"{f['error']}", flush=True)
                with open(log, "a") as fh:
                    fh.write(f"TIC {tic} branch {f['source']} {f['sectors']}: "
                             f"{f['error']}\n")
        except Exception as exc:  # noqa: BLE001
            print(f"{tag}: FAILED  {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            with open(log, "a") as fh:
                fh.write(f"TIC {tic} FAILED: {type(exc).__name__}: {exc}\n")
                fh.write(traceback.format_exc() + "\n")

    print("\n" + "=" * 78)
    print(f"{'TIC':>10} {'Tmag':>5} {'sec':>4} {'cadences':>9} {'p2p ppm':>8} "
          f"{'MAD ppm':>8} {'min':>6}")
    for r in summary:
        print(f"{r['tic']:>10} {r['tmag']:5.2f} {r['nsec']:4d} {r['n']:9d} "
              f"{r['p2p']:8.0f} {r['mad']:8.0f} {r['mins']:6.1f}")
    print(f"\ntotal wall time: {(time.time() - t_start) / 60:.1f} min")
    print(f"outputs in: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
