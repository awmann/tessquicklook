"""Plot a light curve produced by the pipeline.

Usage:  python tests/plot_lightcurve.py <TIC> <sector> [outfile.png]
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

from tessquicklook import quicklooktessffi  # noqa: E402
from tessquicklook.idlcompat import point_to_point_scatter as p2p  # noqa: E402
from tessquicklook.spline import keplerspline  # noqa: E402

# Validated palette (dataviz reference instance, light mode).
# Checked with the six-checks validator: worst pair CVD dE 24.7, normal 33.6,
# both series >= 3:1 contrast on the surface.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8880"
RAW = "#2a78d6"      # categorical slot 1
COR = "#eb6834"      # categorical slot 2
GRID = "#e5e4df"
FLAG = "#efe7d6"     # neutral annotation band, not a data series

GAP_DAYS = 0.05


def _break_gaps(t, y, gap=GAP_DAYS):
    """Insert NaNs across time gaps so lines are not drawn through them."""
    order = np.argsort(t)
    t, y = t[order], y[order]
    idx = np.where(np.diff(t) > gap)[0]
    for k, i in enumerate(idx):
        t = np.insert(t, i + 1 + k, np.nan)
        y = np.insert(y, i + 1 + k, np.nan)
    return t, y


def mad_std(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def main():
    tic = int(sys.argv[1])
    sector = int(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else f"/tmp/tic{tic}_s{sector}.png"

    res = quicklooktessffi(
        tic, xsize=15, ysize=15, corrndays=0.3, usecbv=True, skew=True, kurt=True,
        only_sectors=[sector], rebin=False, verbose=True,
    )
    s = res["sectors"][0]
    t, raw, cor = s["t"], s["f"], s["fcor"]
    t0 = np.floor(t.min())
    x = t - t0

    var, _, _ = keplerspline(t, cor, ndays=0.3)
    vx, vy = _break_gaps(x, var)

    # Region 1: the stretch immediately after the longest gap.  For sector 1751
    # that gap is a 2.81 d safe mode (DRN DR136 §1), so this is post-safe-mode
    # recovery rather than an ordinary start-of-orbit ramp.
    order = np.argsort(t)
    gaps = np.diff(t[order])
    ramp_lo = ramp_hi = None
    if gaps.size and gaps.max() > 1.0:
        i = int(np.argmax(gaps))
        ramp_lo = t[order][i + 1] - t0
        after = t[order][i + 1:]
        nxt = np.where(np.diff(after) > 0.1)[0]
        ramp_hi = (after[nxt[0]] - t0) if nxt.size else ramp_lo + 0.5

    # Region 2: elevated background (scattered light).  Data-driven rather than
    # hard-coded, because SPOC sets no stray-light flag on these FFIs.
    bkg = np.asarray(s["raw"]["medians"], dtype=float)
    floor = np.median(bkg)
    hot = bkg > 2.0 * floor
    sl_lo = (t[hot].min() - t0) if hot.any() else None
    sl_hi = (t[hot].max() - t0) if hot.any() else None

    # This user's matplotlibrc enables constrained_layout, which silently
    # ignores subplots_adjust; let the engine own the spacing.
    fig, axes = plt.subplots(
        2, 1, figsize=(11.5, 7.6), sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [1, 1]},
    )
    fig.patch.set_facecolor(SURFACE)

    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=INK2, labelsize=9, length=3, width=0.8)
        if ramp_lo is not None:
            ax.axvspan(ramp_lo, ramp_hi, color=FLAG, zorder=1, lw=0)
        if sl_lo is not None:
            ax.axvspan(sl_lo, sl_hi, color=FLAG, zorder=1, lw=0)

    # --- Panel 1: raw vs corrected, offset so both are legible --------------
    ax = axes[0]
    offset = 0.055
    ax.scatter(x, raw + offset, s=4, color=RAW, alpha=0.55, linewidths=0, zorder=3,
               label="Raw aperture photometry (offset +0.055)")
    ax.scatter(x, cor, s=4, color=COR, alpha=0.55, linewidths=0, zorder=3,
               label="Corrected (systematics removed)")

    # Direct labels placed inside the axes, anchored to a quiet stretch.
    late = x > x.max() - 1.0
    ax.annotate("raw", (x[late].mean(), np.median(raw[late]) + offset + 0.012),
                color=INK2, fontsize=9.5, ha="center", fontweight="medium", zorder=6)
    ax.annotate("corrected", (x[late].mean(), np.median(cor[late]) - 0.020),
                color=INK2, fontsize=9.5, ha="center", fontweight="medium", zorder=6)

    ax.set_ylabel("Normalised flux  (offset)", color=INK2, fontsize=10)
    # Lower-left is the one quadrant free of both data and the flagged bands.
    ax.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK2,
              handletextpad=0.4, borderaxespad=0.8, markerscale=2.2)
    ax.set_title(f"TIC {tic}  —  Sector {sector}",
                 color=INK, fontsize=13.5, fontweight="semibold", loc="left", pad=30)

    y0 = ax.get_ylim()[0]
    if ramp_lo is not None:
        ax.annotate("post-safe-mode recovery\n(unflagged)",
                    ((ramp_lo + ramp_hi) / 2, y0 + 0.012),
                    color=INK2, fontsize=8.5, ha="center", va="bottom", zorder=6)
    if sl_lo is not None:
        ax.annotate("scattered light\nbkg x%.1f (unflagged)" % (np.max(bkg) / floor),
                    ((sl_lo + sl_hi) / 2, y0 + 0.012),
                    color=INK2, fontsize=8.5, ha="center", va="bottom", zorder=6)

    # --- Panel 2: the delivered light curve + variability model -------------
    ax = axes[1]
    ax.scatter(x, cor, s=5, color=COR, alpha=0.6, linewidths=0, zorder=3,
               label="Corrected flux")
    ax.plot(vx, vy, color=INK, lw=1.6, zorder=4,
            label="Variability model (0.3 d spline)")

    med_err = np.median(s["err_photon"])
    ylo = np.percentile(cor, 1)
    xe = x.min() + 0.10
    ax.errorbar([xe], [ylo], yerr=[med_err], fmt="none", ecolor=INK2,
                elinewidth=1.2, capsize=3, zorder=6)
    ax.annotate(f"median error {med_err*1e6:.0f} ppm", (xe + 0.13, ylo),
                color=INK2, fontsize=8.5, va="center", zorder=6)

    ax.set_xlabel(f"BTJD − {t0:.0f}", color=INK2, fontsize=10, labelpad=6)
    ax.set_ylabel("Normalised flux", color=INK2, fontsize=10)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK2,
              handletextpad=0.4, borderaxespad=0.8, markerscale=2.0)

    cadence = np.median(np.diff(np.sort(t))) * 86400
    clean = np.ones_like(t, dtype=bool)
    if ramp_lo is not None:
        clean &= (t - t0 < ramp_lo) | (t - t0 > ramp_hi)
    clean &= ~hot
    note = (f"{t.size} cadences · {cadence:.0f} s · {np.ptp(t):.2f} d · "
            f"aperture {'circ' if s['usecirc'] else 'prf'}#{s['best']} · "
            f"point-to-point {p2p(cor)*1e6:.0f} ppm · "
            f"MAD scatter {mad_std(cor)*1e6:.0f} ppm all / "
            f"{mad_std(cor[clean])*1e6:.0f} ppm excluding both flagged regions\n"
            f"corrected against spacecraft quaternions only — "
            f"no CBVs archived at MAST for sector {sector}")
    # Provenance sits under the title, not at the foot of the figure, where it
    # would collide with the x-label.
    axes[0].annotate(note, xy=(0, 1.045), xycoords="axes fraction",
                     color=MUTED, fontsize=8.2, va="bottom", ha="left")

    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"\nWrote {out}")

    # Table view, so nothing depends on colour alone.
    print(f"\n    {'quantity':32s}{'value':>14s}")
    rows = [
        ("cadences", f"{t.size}"),
        ("cadence (s)", f"{cadence:.1f}"),
        ("baseline (d)", f"{np.ptp(t):.2f}"),
        ("aperture", f"{'circ' if s['usecirc'] else 'prf'}#{s['best']}"),
        ("point-to-point (ppm)", f"{p2p(cor)*1e6:.0f}"),
        ("MAD scatter, all (ppm)", f"{mad_std(cor)*1e6:.0f}"),
        ("MAD scatter, clean (ppm)", f"{mad_std(cor[clean])*1e6:.0f}"),
        ("clean cadences", f"{int(clean.sum())}"),
        ("median photon err (ppm)", f"{med_err*1e6:.0f}"),
        ("empirical err (ppm)", f"{s['err_empirical']*1e6:.0f}"),
    ]
    for k, v in rows:
        print(f"    {k:32s}{v:>14s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
