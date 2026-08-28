"""Per-sector summary of a batch run, parsed from its log plus the written CSVs.

    python tests/batch_summary.py [logfile] [output-dir]

``output-dir`` must match the ``--out`` the run used, or the CSV sanity pass at
the end silently describes a different batch's products.

Reports per-sector aperture/scatter (the combined point-to-point of a stitched
multi-cadence light curve is not a meaningful number on its own).
"""

import re
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "output" / "lightcurves"


def parse(logpath):
    tic = None
    targets = {}
    tic_re = re.compile(r"^TIC (\d+): RA=([\d.]+) Dec=([-\d.]+) Tmag=([-\d.]+)")
    # The FFI branch reports camera/ccd only; the short-cadence branch appends
    # the exposure time.
    sec_re = re.compile(
        r"^Processing Sector (\d+) \(camera (\d+), ccd (\d+)(?:, (\d+)s)?\)")
    ap_re = re.compile(r"^  aperture: (\S+) #(\d+), scatter (\d+) ppm, (\d+) cadences")
    sc_re = re.compile(r"^  CROWDSAP=([\d.]+)\s+(\d+) regressors\s+"
                       r"scatter (\d+) ppm, (\d+) cadences")
    pending = None
    for line in Path(logpath).read_text(errors="replace").splitlines():
        m = tic_re.match(line)
        if m:
            tic = int(m.group(1))
            targets.setdefault(tic, dict(tmag=float(m.group(4)), sectors=[]))
            continue
        m = sec_re.match(line)
        if m:
            pending = (int(m.group(1)), int(m.group(2)), int(m.group(3)),
                       int(m.group(4)) if m.group(4) else None)
            continue
        if tic is None or pending is None:
            continue

        m = ap_re.match(line)
        if m:
            kind = "circ" if m.group(1).startswith("circ") else "PRF"
            targets[tic]["sectors"].append(dict(
                sector=pending[0], camera=pending[1], ccd=pending[2],
                exptime=pending[3] or 0, label=f"{kind} #{m.group(2)}",
                scatter=int(m.group(3)), n=int(m.group(4))))
            pending = None
            continue

        m = sc_re.match(line)
        if m:
            targets[tic]["sectors"].append(dict(
                sector=pending[0], camera=pending[1], ccd=pending[2],
                exptime=pending[3] or 0, label=f"SAP/{m.group(2)}v",
                scatter=int(m.group(3)), n=int(m.group(4))))
            pending = None
    return targets


def main():
    global OUT

    logpath = sys.argv[1] if len(sys.argv) > 1 else "/tmp/batch_run.log"
    if len(sys.argv) > 2:
        OUT = Path(sys.argv[2])
    targets = parse(logpath)

    print(f"{'TIC':>10} {'Tmag':>5} {'sec':>4} {'cam/ccd':>7} {'cad':>6} "
          f"{'source':>10} {'scatter':>8} {'cadences':>9}")
    print("-" * 70)
    grand = 0
    for tic in sorted(targets, key=lambda k: -len(targets[k]["sectors"])):
        d = targets[tic]
        for j, s in enumerate(d["sectors"]):
            head = f"{tic:>10} {d['tmag']:5.2f}" if j == 0 else " " * 16
            cad = f"{s['exptime']}s" if s["exptime"] else "FFI"
            print(f"{head} {s['sector']:>4} {s['camera']}/{s['ccd']:<5} "
                  f"{cad:>6} {s['label']:>10} {s['scatter']:>6} ppm "
                  f"{s['n']:>9}")
            grand += s["n"]
        print()

    # Reconcile against what the run intended to do.  A sector planned but
    # absent from the log means it dropped out mid-run -- most often because the
    # quaternion engineering file for that sector has not been published yet.
    # A sector offered by TESScut but never planned was vetoed by tess-point as
    # off-detector (see the offcollateral note in the README).
    print("Planned vs actually produced")
    print(f"{'TIC':>10} {'planned':>8} {'done':>5}  missing            "
          f"TESScut-only (off-detector)")
    try:
        import warnings as _w

        _w.filterwarnings("ignore")
        from astropy.coordinates import SkyCoord
        from astroquery.mast import Tesscut

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from tessquicklook import plan_cadences
        from tessquicklook.catalog import query_tic

        for tic in sorted(targets):
            plan = plan_cadences(tic)
            planned = {s for secs in plan.values() for s in secs}
            done = {d["sector"] for d in targets[tic]["sectors"]}

            s = query_tic(tic)
            av = Tesscut.get_sectors(
                coordinates=SkyCoord(s["ra"], s["dec"], unit="deg"))
            offered = {int(x) for x in av["sector"]}

            missing = sorted(planned - done)
            vetoed = sorted(offered - planned)
            print(f"{tic:>10} {len(planned):>8} {len(done):>5}  "
                  f"{str(missing if missing else '-'):<18} {vetoed if vetoed else '-'}")
    except Exception as exc:  # noqa: BLE001
        print(f"  (reconciliation skipped: {type(exc).__name__}: {exc})")
    print()

    csvs = sorted(OUT.glob("TIC*.csv"))
    print(f"{len(targets)} targets, "
          f"{sum(len(d['sectors']) for d in targets.values())} sectors, "
          f"{grand} cadences")
    print(f"{len(csvs)} CSVs written to {OUT}")

    # Sanity pass over the written products.
    print(f"\n{'file':>22} {'rows':>8} {'nan flux':>9} {'|f-1|>0.5':>10}")
    for c in csvs:
        a = np.genfromtxt(c, delimiter=",", names=True)
        f = a["flux"]
        print(f"{c.name:>22} {f.size:>8} {int(np.sum(~np.isfinite(f))):>9} "
              f"{int(np.sum(np.abs(f - 1) > 0.5)):>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
