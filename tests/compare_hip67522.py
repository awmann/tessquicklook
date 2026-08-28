"""Compare the Python FFI pipeline against the IDL pipeline for HIP 67522.

IMPORTANT — these are not the same input data.  The Python light curve is FFI
photometry (30 min / 10 min / 200 s depending on sector); the IDL reference was
run on SPOC target-pixel files (2 min / 20 s).  Both are compared after binning
to a common 30-minute grid, so differences reflect *pipeline + data source*
together, not the pipeline alone.

    python tests/compare_hip67522.py
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

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880"
PY, IDL, GRID = "#eb6834", "#2a78d6", "#e5e4df"

REPO = Path(__file__).resolve().parents[1]
PY_CSV = REPO / "output" / "lightcurves" / "TIC166527623.csv"


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


IDL_DIR = _reference_dir("HIP67522_v3", "data")
OUT = REPO / "output"

BIN_MIN = 30.0
P_ROT = 1.418          # HIP 67522 rotation period, days
TREND_DAYS = 3 * P_ROT  # running-median window for the slow/fast split
SECTORS = [11, 38, 64, 101, 102]


def mad_std(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def p2p(x):
    return np.median(np.abs(np.diff(x))) * 1.48 / np.sqrt(2) if x.size > 1 else np.nan


def binned(t, f, width_days, grid_origin):
    """Bin onto a fixed grid so both curves land in identical bins."""
    idx = np.floor((t - grid_origin) / width_days).astype(int)
    out_t, out_f, out_n = [], [], []
    for i in np.unique(idx):
        m = idx == i
        # >=1, not >=2: sector 11's FFI cadence already equals the bin width,
        # so requiring two points per bin would discard the whole sector.
        if m.sum() >= 1:
            out_t.append(grid_origin + (i + 0.5) * width_days)
            out_f.append(np.mean(f[m]))
            out_n.append(m.sum())
    return np.array(out_t), np.array(out_f), np.array(out_n)


def main():
    py = np.genfromtxt(PY_CSV, delimiter=",", names=True)
    tp, fp = py["time"], py["flux"]

    idl = {}
    for s in SECTORS:
        f = IDL_DIR / f"lc_TESS{s}.dat"
        if f.exists():
            d = np.loadtxt(f)
            idl[s] = (d[:, 0], d[:, 1])

    width = BIN_MIN / 60.0 / 24.0
    rows = []
    panels = []

    for s in SECTORS:
        if s not in idl:
            continue
        ti, fi = idl[s]
        lo, hi = ti.min(), ti.max()
        m = (tp >= lo - 0.5) & (tp <= hi + 0.5)
        if m.sum() < 50:
            continue
        origin = np.floor(min(lo, tp[m].min()))

        bt_p, bf_p, _ = binned(tp[m], fp[m], width, origin)
        bt_i, bf_i, _ = binned(ti, fi, width, origin)

        # keep only bins present in both
        common, ip, ii = np.intersect1d(
            np.round(bt_p, 6), np.round(bt_i, 6), return_indices=True)
        if common.size < 30:
            continue
        a, b = bf_p[ip], bf_i[ii]

        r = np.corrcoef(a, b)[0, 1]
        diff = a - b

        # Separate "same astrophysics" from "different long-term systematics":
        # subtract a running median from each, leaving the rotation signal and
        # flares.  The window must be several rotation periods or it eats the
        # rotation itself -- HIP 67522 has P_rot = 1.418 d, so a 1-day window
        # clips the sinusoid badly; TREND_DAYS = 3 * P_rot preserves it.
        from scipy.ndimage import median_filter
        win = max(int(round(TREND_DAYS / (BIN_MIN / 60.0 / 24.0))), 3)
        ah = a - median_filter(a, size=win, mode="nearest")
        bh = b - median_filter(b, size=win, mode="nearest")
        r_hp = np.corrcoef(ah, bh)[0, 1]
        rms_hp = np.std(ah - bh)
        # and the slow part alone
        r_lo = np.corrcoef(a - ah, b - bh)[0, 1]
        rms_lo = np.std((a - ah) - (b - bh))
        # amplitude ratio via regression through the origin on (f-1)
        slope = np.sum((a - 1) * (b - 1)) / np.sum((b - 1) ** 2)

        rows.append(dict(
            sector=s, n=common.size,
            p2p_py=p2p(fp[m]) * 1e6, p2p_idl=p2p(fi) * 1e6,
            p2p_py_b=p2p(a) * 1e6, p2p_idl_b=p2p(b) * 1e6,
            amp_py=mad_std(a) * 1e6, amp_idl=mad_std(b) * 1e6,
            corr=r, rms=np.std(diff) * 1e6, med=np.median(np.abs(diff)) * 1e6,
            slope=slope, r_hp=r_hp, rms_hp=rms_hp * 1e6,
            r_lo=r_lo, rms_lo=rms_lo * 1e6,
            cad_py=np.median(np.diff(np.sort(tp[m]))) * 1440,
            cad_idl=np.median(np.diff(ti)) * 1440,
        ))
        panels.append((s, common, a, b, ah, bh))

    # ---------------- metrics table ----------------
    print(f"HIP 67522 / TIC 166527623 — Python FFI vs IDL TPF, "
          f"both binned to {BIN_MIN:.0f} min\n")
    print(f"{'sec':>4} {'cadence py/idl':>15} {'bins':>5} "
          f"{'p2p native py/idl':>20} {'p2p binned py/idl':>20}")
    for r in rows:
        print(f"{r['sector']:>4} {r['cad_py']:6.1f}/{r['cad_idl']:<8.2f} "
              f"{r['n']:>5} {r['p2p_py']:9.0f}/{r['p2p_idl']:<10.0f} "
              f"{r['p2p_py_b']:9.0f}/{r['p2p_idl_b']:<10.0f}")
    print()
    print(f"{'sec':>4} {'corr':>8} {'rms diff':>10} {'med|diff|':>10} "
          f"{'amp py/idl':>18} {'ampl. ratio':>12}")
    for r in rows:
        print(f"{r['sector']:>4} {r['corr']:8.5f} {r['rms']:9.0f}p {r['med']:9.0f}p "
              f"{r['amp_py']:8.0f}/{r['amp_idl']:<9.0f} {r['slope']:12.4f}")

    print()
    print(f"{'sec':>4} {'raw corr':>9} {'HIGH-PASS corr':>15} {'hp rms':>9} "
          f"{'slow-trend corr':>16} {'slow rms':>9}")
    for r in rows:
        print(f"{r['sector']:>4} {r['corr']:9.4f} {r['r_hp']:15.4f} "
              f"{r['rms_hp']:8.0f}p {r['r_lo']:16.4f} {r['rms_lo']:8.0f}p")

    allp = np.concatenate([p[2] for p in panels])
    alli = np.concatenate([p[3] for p in panels])
    print(f"\nALL: corr={np.corrcoef(allp, alli)[0,1]:.5f}  "
          f"rms diff={np.std(allp-alli)*1e6:.0f} ppm  "
          f"median|diff|={np.median(np.abs(allp-alli))*1e6:.0f} ppm  "
          f"amplitude ratio={np.sum((allp-1)*(alli-1))/np.sum((alli-1)**2):.4f}")

    # ---------------- figure 1: per-sector overlay ----------------
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.15 * n + 1.4),
                             layout="constrained", squeeze=False)
    axes = axes[:, 0]
    fig.patch.set_facecolor(SURFACE)
    for ax, (s, t, a, b, _ah, _bh) in zip(axes, panels):
        ax.plot(t, b, lw=1.0, color=IDL, alpha=0.9, zorder=3)
        ax.plot(t, a, lw=1.0, color=PY, alpha=0.9, zorder=4)
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for sd in ("top", "right"):
            ax.spines[sd].set_visible(False)
        for sd in ("left", "bottom"):
            ax.spines[sd].set_color(MUTED)
            ax.spines[sd].set_linewidth(0.8)
        ax.tick_params(colors=INK2, labelsize=8.5, length=3, width=0.8)
        ax.set_ylabel("norm. flux", color=INK2, fontsize=9)
        rr = [r for r in rows if r["sector"] == s][0]
        ax.annotate(f"Sector {s}   r={rr['corr']:.4f}   "
                    f"rms diff {rr['rms']:.0f} ppm",
                    xy=(0.005, 1.02), xycoords="axes fraction",
                    color=INK2, fontsize=9, va="bottom")
    axes[-1].set_xlabel("Time (BTJD)", color=INK2, fontsize=10)
    h = [plt.Line2D([], [], color=PY, lw=2), plt.Line2D([], [], color=IDL, lw=2)]
    fig.legend(h, ["Python (FFI photometry)", "IDL (SPOC TPF photometry)"],
               loc="outside upper right", frameon=False, fontsize=9,
               labelcolor=INK2, ncol=2)
    fig.suptitle(f"HIP 67522 — Python vs IDL, binned to {BIN_MIN:.0f} min",
                 color=INK, fontsize=13, fontweight="semibold", x=0.008, ha="left")
    fig.savefig(OUT / "hip67522_compare_sectors.png", dpi=140, facecolor=SURFACE)
    plt.close(fig)

    # ---------------- figure 2: raw vs high-pass agreement ----------------
    allph = np.concatenate([p[4] for p in panels])
    allih = np.concatenate([p[5] for p in panels])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.ravel():
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for sd in ("top", "right"):
            ax.spines[sd].set_visible(False)
        for sd in ("left", "bottom"):
            ax.spines[sd].set_color(MUTED)
            ax.spines[sd].set_linewidth(0.8)
        ax.tick_params(colors=INK2, labelsize=9, length=3, width=0.8)

    ax = axes[0, 0]
    ax.scatter(alli, allp, s=4, color=PY, alpha=0.3, linewidths=0, zorder=3)
    lim = [min(alli.min(), allp.min()), max(alli.max(), allp.max())]
    ax.plot(lim, lim, color=INK, lw=1.4, zorder=5)
    ax.set_xlabel("IDL flux", color=INK2, fontsize=10)
    ax.set_ylabel("Python flux", color=INK2, fontsize=10)
    ax.set_title(f"As delivered:  r = {np.corrcoef(allp, alli)[0,1]:.4f}",
                 color=INK, fontsize=10.5, loc="left")

    ax = axes[0, 1]
    ax.scatter(allih * 1e6, allph * 1e6, s=4, color=PY, alpha=0.3,
               linewidths=0, zorder=3)
    lim = [min(allih.min(), allph.min()) * 1e6, max(allih.max(), allph.max()) * 1e6]
    ax.plot(lim, lim, color=INK, lw=1.4, zorder=5)
    ax.set_xlabel("IDL, slow trend removed (ppm)", color=INK2, fontsize=10)
    ax.set_ylabel("Python, slow trend removed (ppm)", color=INK2, fontsize=10)
    ax.set_title(f"Rotation + flares only:  r = "
                 f"{np.corrcoef(allph, allih)[0,1]:.4f}",
                 color=INK, fontsize=10.5, loc="left", fontweight="semibold")

    ax = axes[1, 0]
    tall = np.concatenate([p[1] for p in panels])
    ax.scatter(tall, (allp - alli) * 1e6, s=4, color=MUTED, alpha=0.35,
               linewidths=0, zorder=3, label="as delivered")
    ax.scatter(tall, (allph - allih) * 1e6, s=4, color=PY, alpha=0.5,
               linewidths=0, zorder=4, label="slow trend removed")
    ax.axhline(0, color=INK, lw=1.2, zorder=5)
    ax.set_xlabel("Time (BTJD)", color=INK2, fontsize=10)
    ax.set_ylabel("Python − IDL (ppm)", color=INK2, fontsize=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, markerscale=2.5,
              loc="upper left")
    ax.set_title(f"rms {np.std(allp-alli)*1e6:.0f} → "
                 f"{np.std(allph-allih)*1e6:.0f} ppm",
                 color=INK, fontsize=10.5, loc="left")

    ax = axes[1, 1]
    s_zoom, t_z, a_z, b_z, ah_z, bh_z = panels[-1]
    w = (t_z < t_z.min() + 6)
    ax.plot(t_z[w], ah_z[w] * 1e6, lw=1.3, color=PY, zorder=4)
    ax.plot(t_z[w], bh_z[w] * 1e6, lw=1.3, color=IDL, zorder=3)
    ax.set_xlabel("Time (BTJD)", color=INK2, fontsize=10)
    ax.set_ylabel("slow trend removed (ppm)", color=INK2, fontsize=10)
    ax.set_title(f"Sector {s_zoom}, 6-day zoom", color=INK, fontsize=10.5,
                 loc="left")

    h = [plt.Line2D([], [], color=PY, lw=2), plt.Line2D([], [], color=IDL, lw=2)]
    fig.legend(h, ["Python (FFI)", "IDL (SPOC TPF)"],
               loc="outside upper right", frameon=False, fontsize=9,
               labelcolor=INK2, ncol=2)
    fig.suptitle("HIP 67522 — the pipelines agree on the star, "
                 "differ on slow trends",
                 color=INK, fontsize=13, fontweight="semibold", x=0.006, ha="left")
    fig.savefig(OUT / "hip67522_compare_metrics.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    print(f"\nwrote {OUT/'hip67522_compare_sectors.png'}")
    print(f"wrote {OUT/'hip67522_compare_metrics.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
