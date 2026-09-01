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

P  = 3.0223445
T0 = 1740.21306
dur = 0.0902

d = np.load(f"{WD}/all_sectors_sap_pdc.npz")
t_sap, sap, pdc, qual, sec = d['t'], d['sap'], d['pdc'], d['qual'], d['sec']
good = (qual == 0) & np.isfinite(sap) & np.isfinite(pdc)
t_sap, sap, pdc, sec = t_sap[good], sap[good], pdc[good], sec[good]

# normalize per-sector (each sector independently, standard practice)
sap_n = np.empty_like(sap); pdc_n = np.empty_like(pdc)
for s in np.unique(sec):
    m = sec == s
    sap_n[m] = sap[m] / np.nanmedian(sap[m])
    pdc_n[m] = pdc[m] / np.nanmedian(pdc[m])

df = pd.read_csv((ROOT / "Validation" / "TIC27491137.csv"))
t_cus_all = df['time'].values
cus_all = df['flux'].values
# per-sector normalize using the gap-based segments found earlier
gaps = np.where(np.diff(t_cus_all) > 5)[0]
bounds = [0] + list(gaps+1) + [len(t_cus_all)]
cus_n = np.empty_like(cus_all)
for i in range(len(bounds)-1):
    sl = slice(bounds[i], bounds[i+1])
    cus_n[sl] = cus_all[sl] / np.nanmedian(cus_all[sl])
t_cus, cus = t_cus_all, cus_n

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

# detrend per sector separately (avoid smoothing across gaps) then concat
def detrend_persector(t, f, secid, window=0.75):
    out = np.full_like(f, np.nan)
    for s in np.unique(secid):
        m = secid == s
        out[m] = detrend_fast(t[m], f[m], window)
    return out

sap_res = detrend_persector(t_sap, sap_n, sec)
pdc_res = detrend_persector(t_sap, pdc_n, sec)

# build a sector id for custom pipeline segments
cus_sec = np.empty(len(t_cus), dtype=int)
seg_names = [16, 23, 50, 50, 77, 77]  # from earlier gap analysis (50 and 77 split by downlink gap)
for i in range(len(bounds)-1):
    cus_sec[bounds[i]:bounds[i+1]] = seg_names[i]
cus_res = detrend_persector(t_cus, cus, cus_sec)

def fold(t, f):
    ph = ((t - T0 + P/2) % P) - P/2
    return ph

def bin_phase(ph, f, binmin=5, win=6/24):
    m = np.abs(ph) < win
    ph, f = ph[m], f[m]
    binwidth = binmin/60/24
    order = np.argsort(ph)
    ph, f = ph[order], f[order]
    edges = np.arange(-win, win+binwidth, binwidth)
    idx = np.digitize(ph, edges)
    bx, by, be = [], [], []
    for i in range(1, len(edges)):
        mm = idx == i
        if mm.sum() == 0:
            continue
        bx.append(np.nanmean(ph[mm]))
        by.append(np.nanmean(f[mm]))
        be.append(np.nanstd(f[mm])/np.sqrt(mm.sum()))
    return np.array(bx), np.array(by), np.array(be)

ph_sap = fold(t_sap, sap_res)
ph_pdc = fold(t_sap, pdc_res)
ph_cus = fold(t_cus, cus_res)

bx_sap, by_sap, be_sap = bin_phase(ph_sap, sap_res)
bx_pdc, by_pdc, be_pdc = bin_phase(ph_pdc, pdc_res)
bx_cus, by_cus, be_cus = bin_phase(ph_cus, cus_res)

n_transits = (t_cus.max()-t_cus.min())/P
print(f"~{n_transits:.0f} orbits of planet e spanned by full baseline (not all observed)")

def depth_snr(bx, by, be, halfdur):
    intr = np.abs(bx) < halfdur
    oot = ~intr
    depth = np.nanmean(by[oot]) - np.nanmean(by[intr])
    oot_std = np.nanstd(by[oot])
    err_on_mean = oot_std/np.sqrt(intr.sum())
    snr = depth/err_on_mean if err_on_mean>0 else np.nan
    return depth*1e6, err_on_mean*1e6, snr, intr.sum()

print("\nAll 4 sectors (16,23,50,77) combined, phase-folded on planet e ephemeris:")
for name, bx, by, be in [("SAP", bx_sap, by_sap, be_sap),
                          ("PDCSAP", bx_pdc, by_pdc, be_pdc),
                          ("custom", bx_cus, by_cus, be_cus)]:
    depth, e, snr, nbin = depth_snr(bx, by, be, dur/2)
    print(f"{name:8s}: depth = {depth:7.1f} +/- {e:6.1f} ppm  SNR~{snr:5.2f}  ({nbin} in-transit 5-min bins)")

fig, axes = plt.subplots(3, 1, figsize=(9, 11), constrained_layout=True, sharex=True)
labels = ['SAP', 'PDCSAP', 'custom (tessquicklook)']
colors = ['green', 'darkorange', 'purple']
data = [(bx_sap, by_sap, be_sap), (bx_pdc, by_pdc, be_pdc), (bx_cus, by_cus, be_cus)]
for ax, (bx, by, be), lab, c in zip(axes, data, labels, colors):
    ax.errorbar(bx*24, by, yerr=be, fmt='o', ms=4, color=c, alpha=0.85)
    ax.axvspan(-dur/2*24, dur/2*24, color='teal', alpha=0.15)
    ax.axhline(1.0, color='gray', lw=0.7)
    ax.set_ylabel('binned residual flux')
    ax.set_title(f'{lab}  (all 4 sectors, 0.75-day detrend)')
axes[-1].set_xlabel('hours from predicted mid-transit of TOI-2076 e')
fig.savefig(f"{WD}/toi2076_e_allsectors_comparison.png", dpi=150)
print("saved plot: toi2076_e_allsectors_comparison.png")
