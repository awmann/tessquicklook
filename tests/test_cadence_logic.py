"""Offline checks on cadence selection and the short-cadence quality mask.

    python tests/test_cadence_logic.py

No network, no IDL: this covers the pure logic that decides *which* data a run
uses, which is the part most likely to change silently.
"""

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tessquicklook.dispatch import _resolve_priority  # noqa: E402
from tessquicklook.scpipeline import _quality_mask  # noqa: E402
from tessquicklook.spoc import (  # noqa: E402
    EXPTIME_FAST,
    EXPTIME_SHORT,
    _filter_to_target,
    _is_same_tic,
    normalise_exptime,
)

FAILS = []


def check(name, got, want):
    ok = got == want
    if not ok:
        FAILS.append(name)
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:<52} {got!r}")


def test_exptime_aliases():
    print("exposure-time aliases")
    for alias in ("fast", "20s", "20", "twentysecond", 20, 20.0):
        check(f"normalise_exptime({alias!r})", normalise_exptime(alias), EXPTIME_FAST)
    for alias in ("short", "120s", "2min", "2m", 120):
        check(f"normalise_exptime({alias!r})", normalise_exptime(alias), EXPTIME_SHORT)
    try:
        normalise_exptime("600s")
        check("bad alias raises", False, True)
    except ValueError:
        check("bad alias raises ValueError", True, True)


def test_same_tic():
    print("\ntarget identity matching")
    for name in ("56658270", " 56658270 ", "TIC 56658270", "TIC_56658270",
                 "tic 56658270", 56658270):
        check(f"_is_same_tic({name!r})", _is_same_tic(name, 56658270), True)
    for name in ("56658273", "", "not-a-tic", None):
        check(f"_is_same_tic({name!r}) rejected", _is_same_tic(name, 56658270), False)


def test_cone_search_filter():
    """Regression: a cone search returns neighbours, and they must be dropped.

    lk.search_lightcurve("TIC <id>") is positional, so TIC 56658270 also
    returns TIC 56658273 -- at distance 0.0 arcsec, so only the identity check
    separates them.  Unfiltered this doubled that target's light curve from
    31,671 to 63,343 cadences with no warning.
    """
    print("\ncone-search contamination filter")
    from astropy.table import Table

    mixed = Table({"target_name": ["56658270", "56658273", "56658270", "56658273"],
                   "sequence_number": [43, 43, 44, 44]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mask = _filter_to_target(mixed, 56658270)
    check("neighbour rows dropped", list(mask), [True, False, True, False])
    check("warns about the neighbour",
          any("56658273" in str(w.message) for w in caught), True)

    clean = Table({"target_name": ["166527623"] * 3, "sequence_number": [11, 38, 64]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mask = _filter_to_target(clean, 166527623)
    check("uncontaminated target untouched", list(mask), [True, True, True])
    check("and stays silent", len(caught), 0)

    # No match at all -> return nothing, loudly, rather than a neighbour's data
    wrong = Table({"target_name": ["99999999"], "sequence_number": [7]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mask = _filter_to_target(wrong, 56658270)
    check("no match returns empty mask", list(mask), [False])
    check("no match warns", len(caught), 1)

    check("empty table is safe", list(_filter_to_target(Table({"target_name": []}), 1)), [])

    # A table without the column (older lightkurve) must not be filtered away
    nocol = Table({"sequence_number": [11, 38]})
    check("missing target_name column keeps all",
          list(_filter_to_target(nocol, 166527623)), [True, True])


def test_priority():
    print("\ncadence priority resolution")
    check("auto", _resolve_priority("auto"), [EXPTIME_FAST, EXPTIME_SHORT, "ffi"])
    check("None is auto", _resolve_priority(None), [EXPTIME_FAST, EXPTIME_SHORT, "ffi"])
    check("fast", _resolve_priority("fast"), [EXPTIME_FAST])
    check("ffi", _resolve_priority("ffi"), ["ffi"])
    check("sc never falls back to FFI",
          _resolve_priority("sc"), [EXPTIME_FAST, EXPTIME_SHORT])
    check("explicit list keeps order",
          _resolve_priority(["120s", "ffi"]), [EXPTIME_SHORT, "ffi"])
    try:
        _resolve_priority("hourly")
        check("bad cadence raises", False, True)
    except ValueError:
        check("bad cadence raises ValueError", True, True)


def test_quality_mask():
    """The IDL keeps quality 0 and 32768, and nothing else, by default."""
    print("\nshort-cadence quality mask (quicklooksector3.pro)")
    q = np.array([0, 32768, 2048, 4096, 8192, 64, 1024, 0])
    lc = {
        "t": np.arange(q.size, dtype=float),
        "sap": np.ones(q.size),
        "quality": q,
    }
    lc["sap"][-1] = np.nan  # a finite-flux failure that quality alone misses

    m = _quality_mask(lc)
    check("default keeps 0 and 32768 only", m.tolist(),
          [True, True, False, False, False, False, False, False])

    m = _quality_mask(lc, allowscatteredlight=True)
    check("allowscattered adds 2048 and 4096, drops 32768", m.tolist(),
          [True, False, True, True, False, False, False, False])

    quats = {"dataexist": np.array([1, 0, 1, 1, 1, 1, 1, 1])}
    m = _quality_mask(lc, quats=quats)
    check("dataexist=0 vetoes a cadence", m.tolist(),
          [True, False, False, False, False, False, False, False])

    m = _quality_mask(lc, noexclude=True)
    check("noexclude keeps everything finite", m.tolist(),
          [True] * 7 + [False])

    # dataexist must gate the readmitted straylight cadences too.  The IDL
    # scopes it to the `quality eq 0` term, letting cadences with no quaternion
    # coverage in with all-zero regressors.
    quats = {"dataexist": np.array([1, 1, 0, 1, 1, 1, 1, 1])}
    m = _quality_mask(lc, quats=quats, allowscatteredlight=True)
    check("dataexist gates straylight cadences too", m.tolist(),
          [True, False, False, True, False, False, False, False])

    lc2 = dict(lc, sap=np.array([1., 1., np.nan, 1., 1., 1., 1., 1.]))
    m = _quality_mask(lc2, allowscatteredlight=True)
    check("straylight cadence with NaN flux is dropped", bool(m[2]), False)


def main():
    test_exptime_aliases()
    test_same_tic()
    test_cone_search_filter()
    test_priority()
    test_quality_mask()
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {FAILS}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
