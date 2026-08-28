"""``quicklooktess`` -- pick the best available cadence, per sector.

TESS delivers the same star at up to three cadences, and which ones exist
changes sector by sector: 20 s SPOC light curves only for targets on the
fast-cadence list, 120 s SPOC light curves for the (much larger) 2-minute
target list, and FFIs for everything on silicon.  HIP 67522, for instance, has
120 s data in Sectors 11 and 38 but 20 s data in 64, 101 and 102.

This module resolves that per sector rather than per target, then runs the
short-cadence pipeline (:func:`~tessquicklook.scpipeline.quicklooktesssc`) and
the FFI pipeline (:func:`~tessquicklook.pipeline.quicklooktessffi`) over their
respective sector lists and stitches the results.  The correction itself is the
same code in both cases, so the products are directly concatenable; the
``cadence_s`` column records which cadence each point came from.
"""

from __future__ import annotations

import warnings

import numpy as np

from .catalog import observed_sectors, query_tic
from .pipeline import quicklooktessffi, write_lightcurve
from .scpipeline import quicklooktesssc
from .spline import keplerspline
from .spoc import EXPTIME_FAST, EXPTIME_SHORT, available_cadences

__all__ = ["quicklooktess", "plan_cadences"]

# Fastest first.  "ffi" is the fallback that always exists.
_PRIORITY = (EXPTIME_FAST, EXPTIME_SHORT, "ffi")

_ALIASES = {
    "auto": None,
    "fast": (EXPTIME_FAST,),
    "20s": (EXPTIME_FAST,),
    "short": (EXPTIME_SHORT,),
    "120s": (EXPTIME_SHORT,),
    "2min": (EXPTIME_SHORT,),
    "sc": (EXPTIME_FAST, EXPTIME_SHORT),
    "spoc": (EXPTIME_FAST, EXPTIME_SHORT),
    "ffi": ("ffi",),
    "slow": ("ffi",),
}


def _resolve_priority(cadence):
    """Turn the ``cadence`` argument into an ordered list of sources to try."""
    if cadence is None or (isinstance(cadence, str) and cadence.lower() == "auto"):
        return list(_PRIORITY)
    if isinstance(cadence, str):
        key = cadence.strip().lower()
        if key not in _ALIASES:
            raise ValueError(
                f"Unrecognised cadence {cadence!r}; expected one of "
                f"{sorted(_ALIASES)} or a list of them."
            )
        return list(_ALIASES[key])
    # An explicit priority list: ["20s", "ffi"] etc.
    out = []
    for item in cadence:
        out.extend(_resolve_priority(item) if isinstance(item, str) else [float(item)])
    return out


def plan_cadences(ticid, cadence="auto", excludesector=None, only_sectors=None,
                  verbose=False):
    """Decide which data source to use for each sector, without running anything.

    Returns ``{source: sorted_sectors}`` where ``source`` is ``20.0``, ``120.0``
    or ``"ffi"``.  Useful on its own to see what a run will do::

        >>> plan_cadences(166527623)                        # doctest: +SKIP
        {20.0: [64, 101, 102], 120.0: [11, 38]}
    """
    priority = _resolve_priority(cadence)
    excludesector = {int(s) for s in (excludesector or [])}
    only_sectors = {int(s) for s in only_sectors} if only_sectors else None

    spoc = available_cadences(ticid) if any(p != "ffi" for p in priority) else {}

    ffi_sectors = set()
    if "ffi" in priority:
        star = query_tic(ticid)
        try:
            secs, _ = observed_sectors(star["ra"], star["dec"])
            ffi_sectors = {int(s) for s in secs}
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Could not predict FFI sectors: {exc}")

    candidates = set(spoc) | ffi_sectors
    if only_sectors is not None:
        candidates &= only_sectors
    candidates -= excludesector

    plan = {}
    for sec in sorted(candidates):
        for source in priority:
            if source == "ffi":
                # An explicit only_sectors is trusted even where tess-point is
                # silent -- that is how special campaigns outside its pointing
                # table (sector 1751) get through.  quicklooktessffi does its
                # own veto anyway.
                if sec in ffi_sectors or only_sectors is not None:
                    plan.setdefault("ffi", []).append(sec)
                    break
            elif any(np.isclose(source, e) for e in spoc.get(sec, ())):
                plan.setdefault(float(source), []).append(sec)
                break
        else:
            if verbose:
                print(f"  sector {sec}: no data at the requested cadence(s); skipped")
    return plan


def quicklooktess(
    ticid,
    cadence="auto",
    excludesector=None,
    only_sectors=None,
    corrndays=None,
    ndays=0.5,
    usecbv=True,
    order=2,
    torder=5,
    variability_basis="spline",
    contamination=True,
    allowscatteredlight=False,
    solver="normal",
    discard_quaternion_fits=False,
    ffi_options=None,
    sc_options=None,
    outfile=None,
    verbose=True,
):
    """Run the quick-look pipeline at the best cadence available in each sector.

    This is the recommended entry point.  It dispatches to
    :func:`~tessquicklook.scpipeline.quicklooktesssc` for sectors with SPOC
    short-cadence data and :func:`~tessquicklook.pipeline.quicklooktessffi` for
    the rest, then stitches the products into one light curve.

    Parameters
    ----------
    cadence
        ``"auto"`` (default)
            Per sector, take 20 s if it exists, else 120 s, else FFI.
        ``"fast"`` / ``"20s"``, ``"short"`` / ``"120s"``, ``"ffi"``
            Force one source; sectors lacking it are dropped.
        ``"sc"`` / ``"spoc"``
            Short cadence only -- 20 s where available, else 120 s, never FFI.
        list
            An explicit fallback order, e.g. ``["120s", "ffi"]`` to prefer
            2-minute data and fall back to FFIs, never using 20 s.
    corrndays
        Variability-basis knot spacing.  Left as ``None`` each pipeline keeps
        its own default (0.3 d for FFI, 0.2 d for short cadence, matching
        ``bulkrunffi.pro`` and ``bulkrunsc.pro``).
    allowscatteredlight
        Readmit cadences flagged for scattered light (the IDL's
        ``/allowscattered``).  See "Scattered light" in the README -- this is
        not simply a looser mask, and it measurably *worsened* agreement with
        the IDL reference on HIP 67522 Sector 101.
    ffi_options, sc_options
        Dicts of extra keywords passed only to the respective pipeline --
        ``xsize``/``ysize``/``skew``/``kurt``/``rebin`` for the FFI path,
        ``rebin``/``rebin_minutes`` for the short-cadence path.

    Returns
    -------
    dict
        Same schema as the individual pipelines, plus ``plan`` (the
        source-to-sectors mapping actually used) and a ``sectors`` list whose
        entries carry their own ``exptime``.
    """
    plan = plan_cadences(ticid, cadence=cadence, excludesector=excludesector,
                         only_sectors=only_sectors, verbose=verbose)
    if not plan:
        raise RuntimeError(
            f"No data for TIC {ticid} at cadence={cadence!r}"
        )

    if verbose:
        pretty = ", ".join(
            f"{'FFI' if k == 'ffi' else f'{k:.0f}s'}: {v}" for k, v in plan.items()
        )
        print(f"Cadence plan for TIC {ticid} -- {pretty}\n")

    shared = dict(ndays=ndays, usecbv=usecbv, order=order, torder=torder,
                  variability_basis=variability_basis, contamination=contamination,
                  allowscatteredlight=allowscatteredlight, solver=solver,
                  discard_quaternion_fits=discard_quaternion_fits,
                  verbose=verbose)
    if corrndays is not None:
        shared["corrndays"] = corrndays

    # Neither branch rebins by default here: an auto run exists precisely to
    # keep the fastest cadence each sector offers.  (The FFI pipeline on its own
    # still rebins sectors >= 27 to ~30 min, as the IDL does.)
    ffi_kw = {**shared, "rebin": False, **(ffi_options or {})}
    sc_kw = {**shared, "rebin": False, **(sc_options or {})}

    results = []
    failures = []
    for source, sectors in plan.items():
        label = "FFI" if source == "ffi" else f"{source:.0f}s"
        try:
            if source == "ffi":
                results.append(quicklooktessffi(
                    ticid, only_sectors=sectors, **ffi_kw
                ))
            else:
                results.append(quicklooktesssc(
                    ticid, exptime=source, only_sectors=sectors, **sc_kw
                ))
        except Exception as exc:  # noqa: BLE001
            msg = f"{label} sectors {sectors} failed: {type(exc).__name__}: {exc}"
            failures.append(dict(source=source, sectors=list(sectors),
                                 error=f"{type(exc).__name__}: {exc}"))
            warnings.warn(msg)
            # A warning alone is not enough: batch drivers routinely install a
            # blanket filterwarnings("ignore"), and a whole branch disappearing
            # in silence is exactly the failure you must not miss.  `failures`
            # in the returned dict is the durable record.
            if verbose:
                print(f"  BRANCH FAILED -- {msg}")

    if not results:
        raise RuntimeError("Every cadence branch failed; see warnings above")

    keys = ("t", "f", "fcor", "fcormed", "err_photon", "err_empirical", "cadence_s")
    merged = {k: np.concatenate([r[k] for r in results]) for k in keys}

    idx = np.argsort(merged["t"])
    for k in keys:
        merged[k] = merged[k][idx]

    # Refit the flattening spline across the stitched series rather than
    # concatenating per-branch splines, which would leave steps at the joins.
    flat, _, _ = keplerspline(merged["t"], merged["fcor"], ndays=ndays)
    with np.errstate(invalid="ignore", divide="ignore"):
        merged["fflat"] = merged["fcor"] / flat

    sectors_out = sorted(
        (s for r in results for s in r["sectors"]), key=lambda s: s["sector"]
    )

    result = {
        "ticid": int(ticid),
        "star": results[0]["star"],
        "spline": flat,
        "sectors": sectors_out,
        "variability_basis": variability_basis,
        "plan": plan,
        "failures": failures,
        "source": "mixed" if len(results) > 1 else results[0].get("source"),
        **merged,
    }

    if outfile:
        write_lightcurve(result, outfile)
        if verbose:
            print(f"Wrote {outfile}")
    return result
