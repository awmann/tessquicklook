"""Are TOI-5423's transits visible in sector 1751?

Compares the single usable epoch of each planet in sector 1751 against the same
planets folded over normal sectors 71+72, at identical 200 s cadence.

Usage:  python tests/plot_transits.py [outfile.png]
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

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880"
UNBIN, BINNED, GRID = "#eb6834", "#2a78d6", "#e5e4df"
FLAG = "#efe7d6"

G = 6.674e-11
RHO = 2772.14
# Joint-fit posteriors, Juliet_runs/TIC47319867_v2/long_run_joint (BTJD).
PLANETS = {
    "p1  (TOI-5423.02)": dict(p=0.0427650692, b=0.4359620012,
                              P=3.3932258935, t0=2502.2825636143),
    "p2  (TOI-5423.01)": dict(p=0.0500577445, b=0.4131335274,
                              P=5.7105779846, t0=2503.3310796433),
}
for q in PLANETS.values():
    aRs = (RHO * G * (q["P"] * 86400) ** 2 / (3 * np.pi)) ** (1 / 3.0)
    q["T14"] = q["P"] / np.pi * np.sqrt((1 + q["p"]) ** 2 - q["b"] ** 2) / aRs
    q["depth"] = q["p"] ** 2


def flatten(t, f):
    """Spline-flatten with the predicted transits masked out."""
    mask = np.zeros(t.size, bool)
    for q in PLANETS.values():
        ph = (t - q["t0"] + q["P"] / 2) % q["P"] - q["P"] / 2
        mask |= np.abs(ph) < q["T14"] * 0.75
    sp, _, _ = keplerspline(t[~mask], f[~mask], ndays=0.5)
    return f / np.interp(t, t[~mask], sp)


def binned(x, y, width):
    edges = np.arange(x.min(), x.max() + width, width)
    idx = np.digitize(x, edges) - 1
    bx, by, be = [], [], []
    for i in range(len(edges) - 1):
        m = idx == i
        if m.sum() >= 3:
            bx.append(x[m].mean())
            by.append(y[m].mean())
            be.append(np.std(y[m], ddof=1) / np.sqrt(m.sum()))
    return np.array(bx), np.array(by), np.array(be)


def box_model(ph, q):
    """Flat-bottomed transit at the expected depth/duration."""
    return np.where(np.abs(ph) < q["T14"] / 2, 1 - q["depth"], 1.0)


def measure(ph, f, q):
    intr = np.abs(ph) < q["T14"] / 2 * 0.85
    base = (np.abs(ph) > q["T14"] / 2 * 1.2) & (np.abs(ph) < q["T14"] * 2.5)
    if intr.sum() < 5 or base.sum() < 20:
        return None
    dep = np.median(f[base]) - np.mean(f[intr])
    err = p2p(f[base]) / np.sqrt(intr.sum())
    return dep, err


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tic47319867_transits.png"

    data = {}
    for sec in (71, 72, 1751):
        r = quicklooktessffi(47319867, xsize=15, ysize=15, corrndays=0.3,
                             usecbv=True, skew=True, kurt=True,
                             only_sectors=[sec], rebin=False, verbose=False)
        s = r["sectors"][0]
        data[sec] = (s["t"], flatten(s["t"], s["fcor"]))

    # This user's matplotlibrc enables constrained_layout, which silently
    # ignores subplots_adjust.  Work with the engine rather than against it:
    # it places the suptitle, per-axes titles and an "outside" legend without
    # collisions on its own.
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), sharey="row",
                             layout="constrained")
    fig.patch.set_facecolor(SURFACE)

    for (name, q), row in zip(PLANETS.items(), axes):
        # --- left: the one usable sector-1751 epoch ------------------------
        t, f = data[1751]
        n = int(np.round((np.median(t) - q["t0"]) / q["P"]))
        best = None
        for k in (n - 2, n - 1, n, n + 1, n + 2):
            tc = q["t0"] + k * q["P"]
            m = np.abs(t - tc) < q["T14"] * 2.5
            if m.sum() > 40 and (np.abs(t - tc) < q["T14"] / 2 * 0.85).sum() > 5:
                best = (k, tc)
                break
        panels = [(axes[list(PLANETS).index(name)][0], "Sector 1751 — single epoch", best)]

        ax, title, bb = panels[0]
        if bb is None:
            ax.text(0.5, 0.5, "no covered epoch", transform=ax.transAxes,
                    ha="center", color=INK2)
        else:
            k, tc = bb
            m = np.abs(t - tc) < q["T14"] * 2.5
            ph = (t[m] - tc) * 24
            ax.scatter(ph, f[m], s=6, color=UNBIN, alpha=0.45, linewidths=0, zorder=3)
            bx, by, be = binned(ph, f[m], 0.5)
            ax.errorbar(bx, by, yerr=be, fmt="o", ms=5, color=BINNED,
                        ecolor=BINNED, elinewidth=1.2, capsize=0, zorder=5)
            g = np.linspace(ph.min(), ph.max(), 400)
            ax.plot(g, box_model(g / 24, q), color=INK, lw=1.6, zorder=6)
            r = measure((t[m] - tc), f[m], q)
            if r:
                ax.set_title(f"{title}   (epoch {k})", color=INK, fontsize=10,
                             loc="left", pad=6)
                ax.annotate(f"measured {r[0]*1e6:.0f} ± {r[1]*1e6:.0f} ppm "
                            f"= {r[0]/r[1]:.1f}σ",
                            xy=(0.03, 0.06), xycoords="axes fraction",
                            color=INK2, fontsize=9)

        # --- right: sectors 71+72 folded -----------------------------------
        ax = axes[list(PLANETS).index(name)][1]
        phs, fs = [], []
        for sec in (71, 72):
            tt, ff = data[sec]
            ph = (tt - q["t0"] + q["P"] / 2) % q["P"] - q["P"] / 2
            m = np.abs(ph) < q["T14"] * 2.5
            phs.append(ph[m] * 24)
            fs.append(ff[m])
        ph = np.concatenate(phs)
        ff = np.concatenate(fs)
        ax.scatter(ph, ff, s=4, color=UNBIN, alpha=0.30, linewidths=0, zorder=3)
        bx, by, be = binned(ph, ff, 0.5)
        ax.errorbar(bx, by, yerr=be, fmt="o", ms=5, color=BINNED, ecolor=BINNED,
                    elinewidth=1.2, capsize=0, zorder=5)
        g = np.linspace(ph.min(), ph.max(), 400)
        ax.plot(g, box_model(g / 24, q), color=INK, lw=1.6, zorder=6)
        r = measure(ph / 24, ff, q)
        ax.set_title("Sectors 71 + 72 — folded", color=INK, fontsize=10,
                     loc="left", pad=6)
        if r:
            ax.annotate(f"measured {r[0]*1e6:.0f} ± {r[1]*1e6:.0f} ppm "
                        f"= {r[0]/r[1]:.1f}σ",
                        xy=(0.03, 0.06), xycoords="axes fraction",
                        color=INK2, fontsize=9)

        for ax in axes[list(PLANETS).index(name)]:
            ax.set_facecolor(SURFACE)
            ax.grid(True, color=GRID, lw=0.7, zorder=0)
            ax.set_axisbelow(True)
            ax.axvspan(-q["T14"] / 2 * 24, q["T14"] / 2 * 24, color=FLAG, zorder=1, lw=0)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(MUTED)
                ax.spines[side].set_linewidth(0.8)
            ax.tick_params(colors=INK2, labelsize=9, length=3, width=0.8)
            ax.set_ylim(1 - 6 * q["depth"], 1 + 5 * q["depth"])

        axes[list(PLANETS).index(name)][0].set_ylabel(
            f"{name}\nnormalised flux", color=INK2, fontsize=9.5)

    for ax in axes[1]:
        ax.set_xlabel("hours from mid-transit", color=INK2, fontsize=10)

    # One legend for the figure; identity is never colour-alone.
    h = [plt.Line2D([], [], marker="o", ls="", color=UNBIN, alpha=0.6, ms=5),
         plt.Line2D([], [], marker="o", ls="", color=BINNED, ms=6),
         plt.Line2D([], [], color=INK, lw=1.6)]
    fig.legend(h, ["200 s cadence", "30 min bins", "expected depth (joint fit)"],
               loc="outside upper right", frameon=False, fontsize=9,
               labelcolor=INK2, ncol=3)

    fig.suptitle("TOI-5423 (TIC 47319867) — are the transits in sector 1751?",
                 color=INK, fontsize=13, fontweight="semibold", x=0.008,
                 ha="left")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
