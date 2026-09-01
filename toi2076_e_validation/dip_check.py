import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
WD = ROOT / "output" / "toi2076_e_validation"
WD.mkdir(parents=True, exist_ok=True)

t_sap = np.load(f"{WD}/sap_time.npy")
sap = np.load(f"{WD}/sap_flux.npy")
pdc = np.load(f"{WD}/pdcsap_flux.npy")
qual = np.load(f"{WD}/sap_qual.npy")
good = (qual == 0) & np.isfinite(sap) & np.isfinite(pdc)
t_sap, sap, pdc = t_sap[good], sap[good], pdc[good]
sap = sap/np.nanmedian(sap); pdc = pdc/np.nanmedian(pdc)

df = pd.read_csv((ROOT / "Validation" / "TIC27491137.csv"))
seg = (df['time'] > 1929.0) & (df['time'] < 1955.0)
t_cus = df['time'].values[seg]
cus = df['flux'].values[seg] / np.nanmedian(df['flux'].values[seg])

# find the 3 rotational minima by eye from the plot (~1931.2, 1938.6, 1946.2)
minima = [1931.2, 1938.6, 1946.2]
for tm in minima:
    print(f"\n--- dip near BTJD {tm} ---")
    for name, t, f in [("SAP", t_sap, sap), ("PDCSAP", t_sap, pdc), ("custom", t_cus, cus)]:
        m = np.abs(t - tm) < 0.3
        if m.sum() == 0:
            print(f"  {name}: no data")
            continue
        fmin = np.nanpercentile(f[m], 2)
        depth_ppm = (1 - fmin) * 1e6
        print(f"  {name:8s}: min flux (2nd pct) = {fmin:.5f}  -> depth below norm = {depth_ppm:7.0f} ppm  (n={m.sum()})")
