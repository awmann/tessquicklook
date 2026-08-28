"""Compare the Python pipeline against an IDL-produced reference light curve.

Usage:  python tests/compare_reference_lc.py <TIC> [sector ...]
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from tessquicklook import quicklooktessffi  # noqa: E402
from tessquicklook.idlcompat import point_to_point_scatter as p2p  # noqa: E402


def _reference_dir(*parts):
    """Locate the IDL-produced reference light curves.

    These are the original author's private products and are not distributed
    with this repository.  Point ``TESSQUICKLOOK_REFERENCE_DIR`` at wherever you
    keep them; the historical ~/Dropbox/Juliet_runs location is the fallback.
    """
    import os
    root = os.environ.get("TESSQUICKLOOK_REFERENCE_DIR")
    root = Path(root) if root else Path.home() / "Dropbox" / "Juliet_runs"
    return root.joinpath(*parts)


REF_DIR = _reference_dir()


def main():
    tic = int(sys.argv[1]) if len(sys.argv) > 1 else 88785435
    want = [int(s) for s in sys.argv[2:]] or None

    ref = pd.read_csv(REF_DIR / f"{tic}.csv")
    tref, fref = ref["time"].values, ref["flux"].values

    res = quicklooktessffi(
        tic, xsize=15, ysize=15, corrndays=0.3, usecbv=True, skew=True, kurt=True,
        allowscatteredlight=True, rebin=False, only_sectors=want, verbose=True,
    )

    print("\n" + "=" * 68)
    print(f"{'':22s} {'Python':>12s} {'IDL ref':>12s}")
    for s in res["sectors"]:
        lo, hi = s["t"].min(), s["t"].max()
        m = (tref >= lo - 0.5) & (tref <= hi + 0.5)
        if m.sum() == 0:
            continue
        print(f"\nSector {s['sector']}  ({int(m.sum())} ref pts, {s['t'].size} python pts)")
        print(f"  {'n points':20s} {s['t'].size:12d} {int(m.sum()):12d}")
        print(f"  {'p2p scatter (ppm)':20s} {p2p(s['fcor'])*1e6:12.0f} {p2p(fref[m])*1e6:12.0f}")
        print(f"  {'std (ppm)':20s} {np.std(s['fcor'])*1e6:12.0f} {np.std(fref[m])*1e6:12.0f}")
        print(f"  {'aperture':20s} {('circ' if s['usecirc'] else 'prf') + '#' + str(s['best']):>12s}"
              f" {'?':>12s}")

        # Point-by-point on the common time grid.
        common_t = tref[m]
        interp = np.interp(common_t, s["t"], s["fcor"], left=np.nan, right=np.nan)
        good = np.isfinite(interp)
        if good.sum() > 10:
            d = interp[good] - fref[m][good]
            r = np.corrcoef(interp[good], fref[m][good])[0, 1]
            print(f"  {'correlation':20s} {r:12.5f}")
            print(f"  {'median |diff| (ppm)':20s} {np.median(np.abs(d))*1e6:12.0f}")
            print(f"  {'rms diff (ppm)':20s} {np.std(d)*1e6:12.0f}")

    print("\n" + "=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
