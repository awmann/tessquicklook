"""Checks that the dilution scene never counts TIC DUPLICATE/ARTIFACT rows.

    python tests/test_tic_neighbours.py           # offline checks
    python tests/test_tic_neighbours.py --online  # also a live MAST case

Bug found 2026-11-17: the cone search kept DUPLICATE rows, so a star with a
TIC twin was counted twice (target share of the aperture capped at 0.5, light
curve over-corrected ~2x), and duplicated neighbours were over-subtracted.
"""

import sys
from pathlib import Path

import numpy as np
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tessquicklook.catalog import clean_tic_neighbours  # noqa: E402

FAILS = []


def check(name, got, want):
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got}" + ("" if ok else f" (want {want})"))
    if not ok:
        FAILS.append(name)


def ids(t):
    return sorted(str(v) for v in t["ID"])


# MAST returns ID as a string and duplicate_id masked/empty for ordinary rows
base = Table(dict(
    ID=["100", "200", "201", "300", "400", "500"],
    disposition=["", "", "DUPLICATE", "ARTIFACT", "SPLIT", "duplicate"],
    duplicate_id=["", "", "200", "", "", "400"],
    Tmag=[10.0, 11.0, 11.0, 14.0, 12.0, 12.0],
))
check("target 100: drop self, DUPLICATE, ARTIFACT; keep SPLIT",
      ids(clean_tic_neighbours(base, 100)), ["200", "400"])

# target's own twin listed as a DUPLICATE of it
twin = Table(dict(ID=["100", "900", "200"], disposition=["", "DUPLICATE", ""],
                  duplicate_id=["", "100", ""], Tmag=[10.0, 10.0, 12.0]))
check("own twin removed", ids(clean_tic_neighbours(twin, 100)), ["200"])

# target ID is itself the DUPLICATE row: its kept twin is the target again
rev = Table(dict(ID=["900", "100", "200"], disposition=["DUPLICATE", "", ""],
                 duplicate_id=["100", "", ""], Tmag=[10.0, 10.0, 12.0]))
check("target is the DUPLICATE: kept twin removed", ids(clean_tic_neighbours(rev, 900)), ["200"])

# numeric ID columns (e.g. a VizieR table) and float-looking ids
num = Table(dict(ID=np.array([100, 200, 201], dtype=np.int64), disposition=["", "", "DUPLICATE"],
                 duplicate_id=np.ma.masked_array([0.0, 0.0, 200.0], mask=[True, True, False])))
check("numeric ids", ids(clean_tic_neighbours(num, 100)), ["200"])

# no disposition column: only the target is removed (old behaviour)
plain = Table(dict(ID=["100", "200", "201"], Tmag=[10.0, 11.0, 11.0]))
check("no disposition column", ids(clean_tic_neighbours(plain, 100)), ["200", "201"])

check("empty table", len(clean_tic_neighbours(base[:0], 100)), 0)

if "--online" in sys.argv:
    from tessquicklook.catalog import query_tic, query_tic_cone
    # TIC 353053003 (T=10.7) has a DUPLICATE twin TIC 2025009754 0.08" away
    s = query_tic(353053003)
    cat = query_tic_cone(s["ra"], s["dec"], 150.0)
    raw = set(str(v) for v in cat["ID"])
    clean = set(str(v) for v in clean_tic_neighbours(cat, 353053003)["ID"])
    check("live: twin in raw cone", "2025009754" in raw, True)
    check("live: twin removed", "2025009754" in clean, False)
    check("live: target removed", "353053003" in clean, False)
    check("live: no DUPLICATE left",
          int(np.isin(np.asarray(cat["disposition"]).astype(str), ["DUPLICATE"])[
              np.isin(np.asarray(cat["ID"]).astype(str), list(clean))].sum()), 0)

print("FAILED: " + ", ".join(FAILS) if FAILS else "all passed")
sys.exit(1 if FAILS else 0)
