"""Download the raw SAP/PDCSAP light curves used by the other scripts here.

Run this first if `output/toi2076_e_validation/*.npy` and `all_sectors_sap_pdc.npz`
aren't already present (they're git-ignored, reproducible cache). Requires
lightkurve.
"""
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import numpy as np
import lightkurve as lk

ROOT = Path(__file__).resolve().parents[1]
WD = ROOT / "output" / "toi2076_e_validation"
WD.mkdir(parents=True, exist_ok=True)

SECTORS = [16, 23, 50, 77]


def fetch_sector(sector):
    sr = lk.search_lightcurve('TIC 27491137', mission='TESS', author='SPOC', sector=sector)
    if len(sr) == 0:
        raise RuntimeError(f"no SPOC light curve found for sector {sector}")
    lc = sr.download(download_dir=str(WD / "_mast_cache"))
    t = np.asarray(lc['time'].value)
    return t, np.asarray(lc['sap_flux']), np.asarray(lc['pdcsap_flux']), np.asarray(lc['quality'])


def main():
    all_t, all_sap, all_pdc, all_qual, all_sec = [], [], [], [], []
    for sector in SECTORS:
        print(f"sector {sector} ...")
        t, sap, pdc, qual = fetch_sector(sector)
        all_t.append(t); all_sap.append(sap); all_pdc.append(pdc); all_qual.append(qual)
        all_sec.append(np.full(len(t), sector))
        if sector == 23:
            # dip_check.py / make_fig1.py want Sector 23 alone as flat .npy files
            np.save(WD / "sap_time.npy", t)
            np.save(WD / "sap_flux.npy", sap)
            np.save(WD / "pdcsap_flux.npy", pdc)
            np.save(WD / "sap_qual.npy", qual)

    np.savez(WD / "all_sectors_sap_pdc.npz",
              t=np.concatenate(all_t), sap=np.concatenate(all_sap),
              pdc=np.concatenate(all_pdc), qual=np.concatenate(all_qual),
              sec=np.concatenate(all_sec))
    print(f"wrote cache to {WD}")


if __name__ == "__main__":
    main()
