import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
WD = ROOT / "output" / "toi2076_e_validation"
WD.mkdir(parents=True, exist_ok=True)

idl = pd.read_csv(ROOT / "IDLversions" / "27491137_idl.csv")
py = pd.read_csv(ROOT / "Validation" / "TIC27491137.csv")

t_idl, f_idl, c_idl = idl['time'].values, idl['flux'].values, idl['cad'].values
t_py, f_py, c_py = py['time'].values, py['flux'].values, py['cadence_s'].values

sectors = [(1738, 1764, 'Sector 16 (120s)'),
           (1929, 1955, 'Sector 23 (120s)'),
           (2665, 2692, 'Sector 50 (IDL 60s / python 20s)'),
           (3395, 3424, 'Sector 77 (IDL 60s / python 20s)')]

fig, axes = plt.subplots(4, 1, figsize=(11, 14), constrained_layout=True)

print(f"{'sector':32s} {'n_idl':>7s} {'n_py':>7s} {'p2p_idl(ppm)':>13s} {'p2p_py(ppm)':>12s} {'RMS resid on common grid(ppm)':>30s} {'corr':>6s}")

from scipy.interpolate import interp1d

for ax, (lo, hi, label) in zip(axes, sectors):
    mi = (t_idl >= lo) & (t_idl <= hi)
    mp = (t_py >= lo) & (t_py <= hi)
    ti, fi = t_idl[mi], f_idl[mi]
    tp, fp = t_py[mp], f_py[mp]

    oi = np.argsort(ti); ti, fi = ti[oi], fi[oi]
    op = np.argsort(tp); tp, fp = tp[op], fp[op]

    ax.plot(tp, fp, '.', color='purple', ms=2, alpha=0.35, label='python (tessquicklook)', zorder=2)
    ax.plot(ti, fi, '.', color='black', ms=2, alpha=0.45, label='IDL', zorder=3)
    ax.set_title(label)
    ax.set_ylabel('normalized flux')
    ax.legend(markerscale=6, fontsize=8, loc='upper right')

    # point-to-point scatter (Kepler/tess robust p2p, simple version)
    def p2p(f):
        d = np.diff(f)
        return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2) * 1e6

    p2p_idl = p2p(fi)
    p2p_py = p2p(fp)

    # common-grid residual: interpolate python onto IDL's (coarser, or comparable) time grid,
    # only within overlapping range and where gap to nearest point is small
    within = (ti > tp.min()) & (ti < tp.max())
    tii = ti[within]; fii = fi[within]
    interp = interp1d(tp, fp, kind='linear', bounds_error=False)
    fp_on_idl = interp(tii)
    # require a nearby python point (within 5 min) to avoid interpolating across real data gaps
    idx_near = np.searchsorted(tp, tii)
    idx_near = np.clip(idx_near, 1, len(tp)-1)
    nearest_gap = np.minimum(np.abs(tii - tp[idx_near]), np.abs(tii - tp[idx_near-1]))
    close = nearest_gap < 5/60/24
    resid = (fii[close] - fp_on_idl[close])
    rms = np.nanstd(resid) * 1e6
    corr = np.corrcoef(fii[close], fp_on_idl[close])[0, 1]

    print(f"{label:32s} {mi.sum():7d} {mp.sum():7d} {p2p_idl:13.0f} {p2p_py:12.0f} {rms:30.0f} {corr:6.3f}")

axes[-1].set_xlabel('Time (BTJD)')
fig.savefig(f"{WD}/toi2076_idl_vs_python_persector.png", dpi=150)
print("\nsaved per-sector overlay plot")
