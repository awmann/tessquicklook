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

# --- planet e ephemeris (Nardiello/etc. paper 2505.06358) ---
P  = 3.0223445
T0 = 1740.21306   # BTJD
dur = 0.0902      # days

# --- load SAP/PDCSAP sector 23 ---
t_sap = np.load(f"{WD}/sap_time.npy")
sap = np.load(f"{WD}/sap_flux.npy")
pdc = np.load(f"{WD}/pdcsap_flux.npy")
qual = np.load(f"{WD}/sap_qual.npy")

good = (qual == 0) & np.isfinite(sap) & np.isfinite(pdc)
t_sap, sap, pdc = t_sap[good], sap[good], pdc[good]
sap = sap / np.nanmedian(sap)
pdc = pdc / np.nanmedian(pdc)

# --- load custom (tessquicklook) pipeline, restrict to sector 23 segment ---
df = pd.read_csv((ROOT / "Validation" / "TIC27491137.csv"))
seg = (df['time'] > 1929.0) & (df['time'] < 1955.0)
t_cus = df['time'].values[seg]
cus = df['flux'].values[seg]
cus = cus / np.nanmedian(cus)

print(f"SAP: {len(t_sap)} pts, PDCSAP: {len(t_sap)} pts, custom: {len(t_cus)} pts")

# --- simple 0.75-day high-pass detrend (rolling median), identical method on all 3 ---
def detrend(t, f, window=0.75):
    f = np.asarray(f, dtype=float)
    trend = np.full_like(f, np.nan)
    # sort by time already assumed
    for i in range(len(t)):
        m = np.abs(t - t[i]) < window/2
        m[i] = False  # exclude self mildly; not critical
        if m.sum() > 5:
            trend[i] = np.nanmedian(f[m])
        else:
            trend[i] = np.nanmedian(f)
    resid = f / trend
    return resid, trend

# vectorized-ish via pandas rolling on uniform grid is much faster; SAP/PDCSAP are ~2min cadence (uniform-ish)
def detrend_fast(t, f, window=0.75):
    f = pd.Series(f, index=t)
    cad = np.nanmedian(np.diff(t))
    npts = max(3, int(round(window/cad)))
    if npts % 2 == 0:
        npts += 1
    trend = f.rolling(window=npts, center=True, min_periods=max(3,npts//4)).median()
    trend = trend.interpolate(limit_direction='both')
    resid = f / trend
    return resid.values, trend.values

sap_res, sap_trend   = detrend_fast(t_sap, sap)
pdc_res, pdc_trend   = detrend_fast(t_sap, pdc)
cus_res, cus_trend   = detrend_fast(t_cus, cus)

# --- phase fold ---
def fold(t, f):
    ph = ((t - T0 + P/2) % P) - P/2
    return ph, f

def bin_phase(ph, f, binmin=5):
    binwidth = binmin/60/24  # days
    order = np.argsort(ph)
    ph, f = ph[order], f[order]
    edges = np.arange(ph.min(), ph.max()+binwidth, binwidth)
    idx = np.digitize(ph, edges)
    bx, by, be = [], [], []
    for i in range(1, len(edges)):
        m = idx == i
        if m.sum() == 0:
            continue
        bx.append(np.nanmean(ph[m]))
        by.append(np.nanmean(f[m]))
        be.append(np.nanstd(f[m])/np.sqrt(m.sum()))
    return np.array(bx), np.array(by), np.array(be)

ph_sap, _ = fold(t_sap, sap_res)
ph_pdc, _ = fold(t_sap, pdc_res)
ph_cus, _ = fold(t_cus, cus_res)

# restrict to +/- 6 hr around transit for the bottom panel
win = 6/24
def restrict(ph, f):
    m = np.abs(ph) < win
    return ph[m], f[m]

ph_sap_w, sap_res_w = restrict(ph_sap, sap_res)
ph_pdc_w, pdc_res_w = restrict(ph_pdc, pdc_res)
ph_cus_w, cus_res_w = restrict(ph_cus, cus_res)

bx_sap, by_sap, be_sap = bin_phase(ph_sap_w, sap_res_w)
bx_pdc, by_pdc, be_pdc = bin_phase(ph_pdc_w, pdc_res_w)
bx_cus, by_cus, be_cus = bin_phase(ph_cus_w, cus_res_w)

n_transits = np.floor((t_cus.max()-t_cus.min())/P)
print(f"~{n_transits:.0f} transits of planet e within Sector 23 baseline")

# --- quantify: depth of central bin(s) vs out-of-transit scatter ---
def depth_snr(bx, by, be, halfdur):
    intr = np.abs(bx) < halfdur
    oot = ~intr
    if intr.sum() == 0 or oot.sum() < 5:
        return np.nan, np.nan, np.nan
    depth = np.nanmean(by[oot]) - np.nanmean(by[intr])
    oot_std = np.nanstd(by[oot])
    err_on_mean = oot_std/np.sqrt(intr.sum())
    snr = depth/err_on_mean if err_on_mean>0 else np.nan
    return depth*1e6, err_on_mean*1e6, snr

for name, bx, by, be in [("SAP", bx_sap, by_sap, be_sap),
                          ("PDCSAP", bx_pdc, by_pdc, be_pdc),
                          ("custom", bx_cus, by_cus, be_cus)]:
    d, e, snr = depth_snr(bx, by, be, dur/2)
    print(f"{name:8s}: binned in-transit depth = {d:7.1f} +/- {e:6.1f} ppm  (SNR~{snr:.2f})")

# --- plot, Figure-1 style ---
fig, axes = plt.subplots(3, 1, figsize=(9, 11), constrained_layout=True)

ax = axes[0]
ax.plot(t_sap, sap, '.', color='green', ms=2, alpha=0.5, label='SAP')
ax.plot(t_sap, pdc, '.', color='darkorange', ms=2, alpha=0.5, label='PDCSAP')
ax.plot(t_cus, cus, '.', color='purple', ms=2, alpha=0.5, label='custom (tessquicklook)')
ax.set_title('Sector 23 raw/corrected light curves')
ax.set_xlabel('Time (BTJD)')
ax.set_ylabel('normalized flux')
ax.legend(markerscale=4, fontsize=8)

ax = axes[1]
ax.plot(t_sap, sap_res, '.', color='green', ms=2, alpha=0.5)
ax.plot(t_sap, pdc_res, '.', color='darkorange', ms=2, alpha=0.4)
ax.plot(t_cus, cus_res, '.', color='purple', ms=2, alpha=0.5)
for k in range(-8, 9):
    tt = T0 + k*P
    if t_cus.min() < tt < t_cus.max():
        ax.axvline(tt, color='teal', lw=0.6, alpha=0.5)
ax.set_title('0.75-day high-pass detrended (teal lines = predicted planet-e transits)')
ax.set_xlabel('Time (BTJD)')
ax.set_ylabel('residual flux')
ax.set_ylim(0.985, 1.015)

ax = axes[2]
ax.errorbar(bx_sap*24, by_sap, yerr=be_sap, fmt='o', ms=3, color='green', alpha=0.7, label='SAP')
ax.errorbar(bx_pdc*24, by_pdc, yerr=be_pdc, fmt='o', ms=3, color='darkorange', alpha=0.7, label='PDCSAP')
ax.errorbar(bx_cus*24, by_cus, yerr=be_cus, fmt='o', ms=3, color='purple', alpha=0.9, label='custom')
ax.axvspan(-dur/2*24, dur/2*24, color='teal', alpha=0.15, label='predicted transit e')
ax.axhline(1.0, color='gray', lw=0.7)
ax.set_xlabel('hours from predicted mid-transit')
ax.set_ylabel('binned (5 min) residual flux')
ax.set_title('Phase-folded on P=3.02234 d, T0=1740.21306 BTJD (Sector 23 only)')
ax.legend(fontsize=8)

fig.savefig(f"{WD}/toi2076_e_fig1_reproduction.png", dpi=150)
print("saved plot")
