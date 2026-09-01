import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
WD = ROOT / "output" / "toi2076_e_validation"
WD.mkdir(parents=True, exist_ok=True)
P_true, T0_true, dur_true = 3.0223445, 1740.21306, 0.0902

# known planets (NASA Exoplanet Archive, BTJD = BJD-2457000)
known = {
    'b': dict(P=10.35504, T0=1805.85425, dur=3.251/24),
    'c': dict(P=21.01447, T0=1834.66150, dur=4.186/24),
    'd': dict(P=35.12803, T0=1837.93630, dur=3.046/24),
}

def mask_known(t, pad=1.5):
    keep = np.ones(len(t), dtype=bool)
    for name, kp in known.items():
        ph = ((t - kp['T0'] + kp['P']/2) % kp['P']) - kp['P']/2
        intr = np.abs(ph) < (kp['dur']/2 * pad)
        keep &= ~intr
    return keep

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

keep_sap = mask_known(t_sap)
keep_cus = mask_known(t_cus)
print(f"masking known b/c/d transits: SAP/PDCSAP {(~keep_sap).sum()}/{len(keep_sap)} pts removed, "
      f"custom {(~keep_cus).sum()}/{len(keep_cus)} pts removed")

t_sap_m, sap_n_m, pdc_n_m, sec_m = t_sap[keep_sap], sap_n[keep_sap], pdc_n[keep_sap], sec[keep_sap]
t_cus_m, cus_m, cus_sec_m = t_cus[keep_cus], cus[keep_cus], cus_sec[keep_cus]

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

print("detrending (known planets masked)...")
sap_res = detrend_persector(t_sap_m, sap_n_m, sec_m)
pdc_res = detrend_persector(t_sap_m, pdc_n_m, sec_m)
cus_res = detrend_persector(t_cus_m, cus_m, cus_sec_m)

durations = np.linspace(0.04, 0.15, 12)
periods = np.linspace(1.6, 9.0, 6000)

results = {}
for name, t, f in [("SAP", t_sap_m, sap_res), ("PDCSAP", t_sap_m, pdc_res), ("custom", t_cus_m, cus_res)]:
    m = np.isfinite(f)
    tt, ff = t[m], f[m]
    err = np.full_like(ff, np.nanstd(ff))
    bls = BoxLeastSquares(tt, ff, err)
    r = bls.power(periods, durations, oversample=5)
    power = r.power
    med, mad = np.median(power), np.median(np.abs(power - np.median(power)))
    sde = (power - med) / (1.4826*mad)
    ibest = np.argmax(power)
    pbest = periods[ibest]
    window = np.abs(periods - P_true) < 0.05
    ipeak_near = np.where(window)[0][np.argmax(power[window])]
    rank = int((power > power[ipeak_near]).sum())
    results[name] = dict(periods=periods, sde=sde, power=power,
                          global_best_P=pbest, global_best_sde=sde[ibest], global_best_depth=r.depth[ibest],
                          near_e_P=periods[ipeak_near], near_e_sde=sde[ipeak_near], near_e_depth=r.depth[ipeak_near],
                          rank=rank)
    print(f"\n{name}:")
    print(f"  global best peak: P={pbest:.4f} d  SDE={sde[ibest]:.2f}  depth={r.depth[ibest]*1e6:.0f} ppm")
    print(f"  peak nearest planet e (3.0223 d): P={periods[ipeak_near]:.4f} d  SDE={sde[ipeak_near]:.2f}  depth={r.depth[ipeak_near]*1e6:.0f} ppm")
    print(f"  rank of that peak among {len(periods)} trial periods: {rank} periods have higher power (is it the global best? {rank==0})")

fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True, sharex=True)
colors = {'SAP':'green','PDCSAP':'darkorange','custom':'purple'}
for ax, name in zip(axes, ["SAP","PDCSAP","custom"]):
    r = results[name]
    ax.plot(r['periods'], r['sde'], color=colors[name], lw=0.7)
    ax.axvline(P_true, color='teal', lw=1.2, ls='--', label='planet e (3.0223 d)')
    ax.set_title(f"{name}: BLS periodogram (known b/c/d masked, 0.75-day detrend)")
    ax.set_ylabel('SDE')
    ax.legend(fontsize=8, loc='upper right')
axes[-1].set_xlabel('trial period (days)')
fig.savefig(f"{WD}/toi2076_bls_periodogram_masked.png", dpi=150)
print("\nsaved masked periodogram plot")
