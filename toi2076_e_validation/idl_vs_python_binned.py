import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

idl = pd.read_csv(ROOT / "IDLversions" / "27491137_idl.csv")
py = pd.read_csv(ROOT / "Validation" / "TIC27491137.csv")

t_idl, f_idl = idl['time'].values, idl['flux'].values
t_py, f_py = py['time'].values, py['flux'].values

sectors = [(2665, 2692, 'Sector 50'), (3395, 3424, 'Sector 77')]

def p2p(f):
    d = np.diff(f)
    return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2) * 1e6

for lo, hi, label in sectors:
    mi = (t_idl >= lo) & (t_idl <= hi)
    mp = (t_py >= lo) & (t_py <= hi)
    ti, fi = t_idl[mi], f_idl[mi]
    tp, fp = t_py[mp], f_py[mp]
    oi = np.argsort(ti); ti, fi = ti[oi], fi[oi]
    op = np.argsort(tp); tp, fp = tp[op], fp[op]

    # bin python 20s onto IDL's own 60s bin centers (3-point average nearest each ti)
    binw = 60/86400
    fp_binned = np.full(len(ti), np.nan)
    for k, tc in enumerate(ti):
        m = np.abs(tp - tc) < binw/2
        if m.sum() > 0:
            fp_binned[k] = np.nanmean(fp[m])
    good = np.isfinite(fp_binned)
    resid = fi[good] - fp_binned[good]
    rms = np.nanstd(resid)*1e6
    corr = np.corrcoef(fi[good], fp_binned[good])[0,1]
    print(f"{label}: n_matched={good.sum()}  IDL p2p={p2p(fi):.0f} ppm  python(binned to 60s) p2p={p2p(fp_binned[good]):.0f} ppm  "
          f"RMS resid={rms:.0f} ppm  corr={corr:.4f}")
