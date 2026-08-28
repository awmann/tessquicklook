"""Compare the Python short-cadence pipeline against the IDL reference.

    python tests/compare_hip67522_sc.py

Unlike ``compare_hip67522.py`` -- which pitted the Python *FFI* product against
these same files and therefore mixed pipeline differences with data-source
differences -- this is like-for-like.  Both sides start from the same SPOC
mission light-curve files at the same cadence (120 s in Sectors 11 and 38, 20 s
in 64/101/102), so cadences can be matched one-to-one on time and any
disagreement is the pipeline alone.

Needs the IDL reference light curves in ~/Dropbox/Juliet_runs/HIP67522_v3/data/,
which are not distributed with this repo.
"""

import sys
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


def _reference_dir(*parts):
    """Locate the IDL-produced reference light curves.

    These are the original author's private products and are not distributed
    with this repository.  Point ``TESSQUICKLOOK_REFERENCE_DIR`` at wherever you
    keep them; the historical ~/Dropbox/Juliet_runs location is the fallback.
    """
    import os
    root = os.environ.get("TESSQUICKLOOK_REFERENCE_DIR")
    root = Path(root) if root else Path.home() / "Dropbox" / "Juliet_runs"
    return root.joinpath(*parts)


TIC = 166527623
P_ROT = 1.418                 # HIP 67522 rotation period, days
TREND_DAYS = 3 * P_ROT        # slow/fast split; must be > P_rot or it eats it
REF = _reference_dir("HIP67522_v3", "data")
SECTORS = [11, 38, 64, 101, 102]
OUT = Path(__file__).resolve().parents[1] / "output"

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880"
PY, IDL, GRID = "#2a78d6", "#eb6834", "#e5e4df"


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8.5, length=3, width=0.8)


def load_reference(sector):
    path = REF / f"lc_TESS{sector}.dat"
    if not path.exists():
        return None
    a = np.loadtxt(path)
    return a[:, 0], a[:, 1]


def match(t_py, f_py, t_idl, f_idl, tol_days):
    """Pair cadences by nearest time, within ``tol_days``."""
    order = np.argsort(t_idl)
    ti, fi = t_idl[order], f_idl[order]
    pos = np.clip(np.searchsorted(ti, t_py), 1, ti.size - 1)
    left = np.abs(t_py - ti[pos - 1]) < np.abs(t_py - ti[pos])
    idx = np.where(left, pos - 1, pos)
    hit = np.abs(t_py - ti[idx]) < tol_days
    return t_py[hit], f_py[hit], fi[idx[hit]]


def highpass(t, y, window_days):
    """Split into fast (rotation + flares) and slow components by binned median."""
    from scipy.ndimage import median_filter

    dt = np.median(np.diff(t))
    win = max(int(round(window_days / dt)) | 1, 3)
    slow = median_filter(y, size=win, mode="nearest")
    return y - slow, slow


def main():
    res = quicklooktess(TIC, cadence="auto", verbose=True)
    by_sector = {s["sector"]: s for s in res["sectors"]}

    rows, panels = [], []
    for sec in SECTORS:
        ref = load_reference(sec)
        s = by_sector.get(sec)
        if ref is None or s is None:
            print(f"sector {sec}: missing ({'ref' if ref is None else 'python'})")
            continue
        t_ref, f_ref = ref
        cad_d = s["exptime"] / 86400.0
        tm, a, b = match(s["t"], s["fcor"], t_ref, f_ref, tol_days=cad_d / 2)
        if tm.size < 100:
            print(f"sector {sec}: only {tm.size} matched cadences; skipping")
            continue

        # Both are median-normalised already; compare about their own medians.
        a = a / np.median(a)
        b = b / np.median(b)
        d = a - b

        ah, alo = highpass(tm, a, TREND_DAYS)
        bh, blo = highpass(tm, b, TREND_DAYS)

        rows.append(dict(
            sector=sec, exptime=s["exptime"],
            n_py=s["t"].size, n_ref=t_ref.size, n_match=tm.size,
            p2p_py=p2p(a) * 1e6, p2p_idl=p2p(b) * 1e6,
            corr=np.corrcoef(a, b)[0, 1],
            rms=np.std(d) * 1e6, med=np.median(np.abs(d)) * 1e6,
            amp_py=np.std(a) * 1e6, amp_idl=np.std(b) * 1e6,
            corr_hp=np.corrcoef(ah, bh)[0, 1], rms_hp=np.std(ah - bh) * 1e6,
            corr_lo=np.corrcoef(alo, blo)[0, 1], rms_lo=np.std(alo - blo) * 1e6,
        ))
        panels.append((sec, tm, a, b, d))

    if not rows:
        print("nothing to compare")
        return 1

    print(f"\n{'sec':>4} {'cad':>5} {'n python':>9} {'n IDL':>8} {'matched':>8} "
          f"{'p2p py/IDL':>16}")
    for r in rows:
        print(f"{r['sector']:>4} {r['exptime']:4.0f}s {r['n_py']:>9} {r['n_ref']:>8} "
              f"{r['n_match']:>8} {r['p2p_py']:8.0f}/{r['p2p_idl']:<7.0f}")

    print(f"\n{'sec':>4} {'corr':>9} {'rms diff':>10} {'med|diff|':>10} "
          f"{'amp py/IDL':>18} {'ratio':>7}")
    for r in rows:
        print(f"{r['sector']:>4} {r['corr']:9.5f} {r['rms']:9.0f}p {r['med']:9.0f}p "
              f"{r['amp_py']:9.0f}/{r['amp_idl']:<8.0f} "
              f"{r['amp_py'] / r['amp_idl']:7.4f}")

    print(f"\n{'sec':>4} {'raw corr':>9} {'HIGH-PASS corr':>15} {'hp rms':>9} "
          f"{'slow corr':>10} {'slow rms':>9}")
    for r in rows:
        print(f"{r['sector']:>4} {r['corr']:9.4f} {r['corr_hp']:15.4f} "
              f"{r['rms_hp']:8.0f}p {r['corr_lo']:10.4f} {r['rms_lo']:8.0f}p")

    alla = np.concatenate([p[2] for p in panels])
    allb = np.concatenate([p[3] for p in panels])
    print(f"\nALL: corr={np.corrcoef(alla, allb)[0, 1]:.5f}  "
          f"rms diff={np.std(alla - allb) * 1e6:.0f} ppm  "
          f"median|diff|={np.median(np.abs(alla - allb)) * 1e6:.0f} ppm  "
          f"amplitude ratio={np.std(alla) / np.std(allb):.4f}")

    # --- figure 1: per-sector overlay -----------------------------------
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(11.5, 2.1 * n + 1.0),
                             layout="constrained", squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, (sec, tm, a, b, _) in zip(axes[:, 0], panels):
        ax.scatter(tm, b, s=1.5, color=IDL, alpha=0.35, linewidths=0,
                   label="IDL", zorder=2)
        ax.scatter(tm, a, s=1.5, color=PY, alpha=0.35, linewidths=0,
                   label="Python", zorder=3)
        style(ax)
        ax.set_ylabel(f"S{sec}", color=INK2, fontsize=9)
    # Outside the axes, so it cannot sit on top of the light curve.
    fig.legend(*axes[0, 0].get_legend_handles_labels(),
               loc="outside upper right", frameon=False, fontsize=9,
               markerscale=6, ncol=2)
    axes[-1, 0].set_xlabel("Time (BTJD)", color=INK2, fontsize=10)
    axes[0, 0].set_title(
        "HIP 67522 — Python short-cadence pipeline vs IDL reference\n"
        "same SPOC input files, matched cadence-for-cadence",
        color=INK, fontsize=11, loc="left", pad=8)
    f1 = OUT / "hip67522_sc_compare_sectors.png"
    fig.savefig(f1, dpi=140, facecolor=SURFACE)
    plt.close(fig)

    # --- figure 2: agreement diagnostics --------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7), layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    secs = [r["sector"] for r in rows]
    x = np.arange(len(secs))

    ax = axes[0, 0]
    ax.bar(x - 0.19, [r["p2p_py"] for r in rows], 0.38, color=PY, label="Python")
    ax.bar(x + 0.19, [r["p2p_idl"] for r in rows], 0.38, color=IDL, label="IDL")
    style(ax)
    ax.set_xticks(x); ax.set_xticklabels([f"S{s}" for s in secs])
    ax.set_ylabel("point-to-point scatter (ppm)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("Photometric noise", color=INK, fontsize=10, loc="left")

    ax = axes[0, 1]
    ax.bar(x - 0.19, [r["corr"] for r in rows], 0.38, color=MUTED, label="raw")
    ax.bar(x + 0.19, [r["corr_hp"] for r in rows], 0.38, color=PY,
           label=f"rotation+flares (<{TREND_DAYS:.1f} d)")
    style(ax)
    ax.set_xticks(x); ax.set_xticklabels([f"S{s}" for s in secs])
    ax.set_ylim(0.9, 1.0025)
    ax.set_ylabel("correlation", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.set_title("Agreement", color=INK, fontsize=10, loc="left")

    # Residuals against time *since each sector started*, so all five sectors
    # are visible at once instead of the recent pair dominating a BTJD axis.
    ax = axes[1, 0]
    shades = ["#0b0b0b", MUTED, "#7aa9e0", PY, IDL]
    for (sec, tm, _, _, d), colour in zip(panels, shades):
        ax.scatter(tm - tm.min(), d * 1e6, s=1, alpha=0.2, linewidths=0,
                   color=colour, label=f"S{sec}")
    style(ax)
    ax.legend(frameon=False, fontsize=8, markerscale=6, ncol=5,
              loc="upper center")
    ax.set_xlabel("days since sector start", color=INK2, fontsize=9)
    ax.set_ylabel("Python − IDL (ppm)", color=INK2, fontsize=9)
    ax.set_title("Residual", color=INK, fontsize=10, loc="left")

    ax = axes[1, 1]
    sec, tm, a, b, _ = panels[-1]
    m = tm < tm.min() + 6
    ax.scatter(tm[m], b[m], s=2, color=IDL, alpha=0.4, linewidths=0, label="IDL")
    ax.scatter(tm[m], a[m], s=2, color=PY, alpha=0.4, linewidths=0, label="Python")
    style(ax)
    ax.set_xlabel("Time (BTJD)", color=INK2, fontsize=9)
    ax.set_ylabel("norm. flux", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, markerscale=5)
    ax.set_title(f"Sector {sec}, first 6 days", color=INK, fontsize=10, loc="left")

    f2 = OUT / "hip67522_sc_compare_metrics.png"
    fig.savefig(f2, dpi=140, facecolor=SURFACE)
    plt.close(fig)

    print(f"\nwrote {f1}\n      {f2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
