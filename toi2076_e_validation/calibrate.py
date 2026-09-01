import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
WD = ROOT / "output" / "toi2076_e_validation"
WD.mkdir(parents=True, exist_ok=True)
P  = 3.0223445
T0 = 1740.21306
dur = 0.0902

d = np.load(f"{WD}/all_sectors_sap_pdc.npz")
t_sap, sap, pdc, qual, sec = d['t'], d['sap'], d['pdc'], d['qual'], d['sec']
good = (qual == 0) & np.isfinite(sap) & np.isfinite(pdc)
t_sap, sap, pdc, sec = t_sap[good], sap[good], pdc[good], sec[good]
sap_n = np.empty_like(sap); pdc_n = np.empty_like(pdc)
for s in np.unique(sec):
    m = sec == s
    sap_n[m] = sap[m] / np.nanmedian(sap[m])
    pdc_n[m] = pdc[m] / np.nanmedian(pdc[m])

df = pd.read_csv((ROOT / "Validation" / "TIC27491137.csv"))
t_cus_all = df['time'].values
cus_all = df['flux'].values
gaps = np.where(np.diff(t_cus_all) > 5)[0]
bounds = [0] + list(gaps+1) + [len(t_cus_all)]
cus_n = np.empty_like(cus_all)
for i in range(len(bounds)-1):
    sl = slice(bounds[i], bounds[i+1])
    cus_n[sl] = cus_all[sl] / np.nanmedian(cus_all[sl])
t_cus, cus = t_cus_all, cus_n
seg_names = [16, 23, 50, 50, 77, 77]
cus_sec = np.empty(len(t_cus), dtype=int)
for i in range(len(bounds)-1):
    cus_sec[bounds[i]:bounds[i+1]] = seg_names[i]

def detrend_fast(t, f, window=0.75):
    order = np.argsort(t)
    t2, f2 = t[order], f[order]
    fser = pd.Series(f2, index=t2)
    cad = np.nanmedian(np.diff(t2[:2000])) if len(t2) > 2000 else np.nanmedian(np.diff(t2))
    npts = max(3, int(round(window/cad)))
    if npts % 2 == 0:
        npts += 1
    trend = fser.rolling(window=npts, center=True, min_periods=max(3, npts//4)).median()
    trend = trend.interpolate(limit_direction='both')
    resid = (fser / trend).values
    out = np.empty_like(resid)
    out[order] = resid
    return out

def detrend_persector(t, f, secid, window=0.75):
    out = np.full_like(f, np.nan)
    for s in np.unique(secid):
        m = secid == s
        out[m] = detrend_fast(t[m], f[m], window)
    return out

print("detrending (cached per dataset)...")
sap_res = detrend_persector(t_sap, sap_n, sec)
pdc_res = detrend_persector(t_sap, pdc_n, sec)
cus_res = detrend_persector(t_cus, cus, cus_sec)

def depth_stat(t, f, period, t0, halfdur, win=6/24, binmin=5):
    ph = ((t - t0 + period/2) % period) - period/2
    m = np.abs(ph) < win
    ph2, f2 = ph[m], f[m]
    if len(ph2) < 50:
        return np.nan
    binwidth = binmin/60/24
    edges = np.arange(-win, win+binwidth, binwidth)
    idx = np.digitize(ph2, edges)
    bx, by = [], []
    for i in range(1, len(edges)):
        mm = idx == i
        if mm.sum() == 0:
            continue
        bx.append(np.nanmean(ph2[mm]))
        by.append(np.nanmean(f2[mm]))
    bx, by = np.array(bx), np.array(by)
    intr = np.abs(bx) < halfdur
    oot = ~intr
    if intr.sum() < 3 or oot.sum() < 10:
        return np.nan
    depth = np.nanmean(by[oot]) - np.nanmean(by[intr])
    err = np.nanstd(by[oot]) / np.sqrt(intr.sum())
    return depth / err if err > 0 else np.nan

# real ephemeris SNR
real = {}
real['SAP'] = depth_stat(t_sap, sap_res, P, T0, dur/2)
real['PDCSAP'] = depth_stat(t_sap, pdc_res, P, T0, dur/2)
real['custom'] = depth_stat(t_cus, cus_res, P, T0, dur/2)
print("real-ephemeris stat:", real)

# null distribution: random periods in [1.5, 8] d (avoid exact P and its low harmonics),
# random T0 offsets, same duration, same pipeline
rng = np.random.default_rng(42)
ntrials = 300
null = {'SAP': [], 'PDCSAP': [], 'custom': []}
trial_periods = []
while len(trial_periods) < ntrials:
    ptrial = rng.uniform(1.5, 8.0)
    if abs(ptrial - P) < 0.05 or abs(ptrial - P/2) < 0.03 or abs(ptrial - 2*P) < 0.05:
        continue
    trial_periods.append(ptrial)

for ptrial in trial_periods:
    t0trial = T0 + rng.uniform(0, ptrial)
    null['SAP'].append(depth_stat(t_sap, sap_res, ptrial, t0trial, dur/2))
    null['PDCSAP'].append(depth_stat(t_sap, pdc_res, ptrial, t0trial, dur/2))
    null['custom'].append(depth_stat(t_cus, cus_res, ptrial, t0trial, dur/2))

for k in null:
    arr = np.array(null[k])
    arr = arr[np.isfinite(arr)]
    mu, sd = np.nanmean(arr), np.nanstd(arr)
    z = (real[k] - mu) / sd
    frac_more_extreme = np.mean(arr >= real[k])
    print(f"{k:8s}: null mean={mu:5.2f} std={sd:5.2f} (n={len(arr)})  "
          f"real={real[k]:5.2f}  z-vs-null={z:5.2f}  frac(null>=real)={frac_more_extreme:.3f}")

np.savez(f"{WD}/null_dist.npz", **{k: np.array(v) for k, v in null.items()}, real=np.array([real['SAP'], real['PDCSAP'], real['custom']]))
