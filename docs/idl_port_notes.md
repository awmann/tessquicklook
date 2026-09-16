# IDL port notes

Background detail for [tessquicklook](../README.md): where and why the port
departs from the IDL pipeline, bugs found in the IDL along the way, the
validation numbers, and data-source quirks. None of this is needed to *use* the
package, but read it before "fixing" behaviour that looks odd.

## Contents

* [Bugs found in the IDL](#bugs-found-in-the-idl)
* [Deliberate deviations](#deliberate-deviations)
* [Validation against IDL](#validation-against-idl)
* [Scattered light: measured effect](#scattered-light-measured-effect)
* [20-second CBVs](#20-second-cbvs)
* [Special sectors: 1751 in detail](#special-sectors-1751-in-detail)

## Bugs found in the IDL

### 1. The simultaneous-spline path is silently disabled (important)

`keplerspline(..., /calcafull, afullout=afull)` is supposed to return the
B-spline **design matrix**, which `decorrelatehr858` column-stacks with the
quaternion/CBV vectors so variability and systematics are solved jointly.

It never comes back. `keplerspline` obtains it via `bspline_iterfit`, and
`bspline_iterfit.pro` line 82 (stock idlutils)
declares `_EXTRA=EXTRA` rather than `_REF_EXTRA`. IDL only propagates *output*
keywords by reference, so `afullout` is discarded in transit. Verified both
ways on IDL 8.5: through `bspline_iterfit` → `afull` undefined; calling the
locally-modified `bspline_fit` directly → returns correctly (22×400).

Consequence: `n_elements(afull) eq 0` is always true, so `decorrelatehr858`
takes its **polynomial** branch — the variability basis becomes a degree-5
polynomial in normalised time, and `corrnd=0.3` has no effect on the correction.

This matters a lot. A degree-5 polynomial cannot track a young star's rotation
across a 24-day sector, so the quaternion vectors absorb the stellar signal and
it is then subtracted. On TIC 88785435 sector 11:

| variability basis | point-to-point | systematics model amplitude |
|---|---|---|
| polynomial (`torder=5`) | 2703 ppm | 48384 ppm (absurd) |
| **spline (`corrndays=0.3`)** | **890 ppm** | **6252 ppm** |
| IDL reference light curve | 892 ppm | — |

The author's archived reference light curves match the *spline*
result, so whichever machine produced them had a working `calcafull`. But the
`idlutils` copy on the machine this was ported from does **not**, so running
the IDL against that tree would silently produce much worse light curves.

This port defaults to `variability_basis="spline"`. Pass `"poly"` to reproduce
the broken-fallback behaviour for diagnosis.

### 2. `gettessprf`'s `/blend` branch reads one file four times

`gettessprf.pro` lines 64-67 assign
`prf11`, `prf12`, `prf21`, `prf22` all from `filename11`. The bilinear weights
sum to 1, so the interpolation collapses to the single nearest grid point.
Harmless in effect — the port just does nearest-neighbour directly — but the
blending it appears to perform is not happening.

### 3. `quicklooksector3` never standardises its background regressor

The short-cadence routine standardises every design column to zero mean and unit
variance at lines 196-201 — but the loop runs `for jj = 0, ncols(vec)-1`, and
`bgvec` has one *more* column than `vec`. Its last column, the flattened
`SAP_BKG`, is therefore skipped: it enters the fit in raw counts while every
other column is unit-variance, on a design matrix that is already
ill-conditioned.

Only the `flux_med` product is affected, not `flux`. This port equilibrates
columns inside the solver (see below), which fixes it as a side effect, so the
Python `flux_med` will not match a faithful IDL run bit-for-bit. That is the
correct behaviour, not a porting error.

## Deliberate deviations

* **Numerical conditioning.** The design matrix is badly scaled: quaternion
  regressors span ~11 orders of magnitude (`q1std` ~5e-7, `q1q3std` ~7e-12,
  `q1skew` ~5e-1), and `order=2` squares that, pushing `cond(X'X)` past 1e46.
  Columns are equilibrated to unit norm before solving, which leaves the fitted
  model and the variability/systematics split mathematically unchanged.
* **`cdpp` is cadence-aware.** The IDL's `naverage = hours*2+1` hard-codes
  30-minute cadence; at 200 s the "6 hour" window is really 43 minutes. Pass
  `idl_cdpp_window=True` for the original behaviour. (In testing this did not
  change the selected aperture, but it is wrong as written.)
* **Ephemeris range.** The bundled `2018-2024_tessephemeris.idl` ends
  2024-04-04. The IDL's range check is commented out, so it extrapolates
  silently; this port falls back to SPOC's `TIMECORR` beyond that and warns.
* **`interpolate(cubic=-0.5)`** is implemented as the Keys cubic *convolution*
  kernel. scipy's `map_coordinates(order=3)` is a cubic *B-spline*, a different
  operator — using it would introduce 0.26% PRF errors.
* **CBVs are matched on `CADENCENO`.** `quicklooksector3.pro` appends
  `cbv.vector_k` to the design matrix positionally, assuming the CBV file has one
  row per light-curve cadence in the same order. That does hold for every file
  checked, but matching on the cadence number is exact regardless, and turns a
  hypothetical misalignment into a warning instead of silently wrong output.
* **`dataexist` is applied under `/allowscattered` too.** Both IDL routines
  apply the `quats.dataexist` requirement only inside the *non*-scattered
  branch, so `/allowscattered` admits cadences with no quaternion coverage.
  Those rows enter the design matrix with every quaternion regressor set to
  zero, which is not neutral in a least-squares fit — it shifts the column
  means used for centring and pulls the fit toward the intercept. Across the
  cached SPOC files, 35,670 of 543,747 straylight-flagged cadences (6.6%) have
  no quaternion coverage, reaching 45% (1,440 of 3,233) in TIC 410214986
  sector 68. This port applies `dataexist` unconditionally, and additionally
  requires finite flux on the readmitted cadences.

## Validation against IDL

Component-level, against reference output generated by real IDL 8.5. The
reference data itself is **not** part of this repository (see
[The IDL original](../README.md#the-idl-original)); these are the numbers it produced:

| component | agreement |
|---|---|
| `keplerspline` | 2.6e-07 (0.001% of model range) |
| B-spline knot vector | 4.8e-13 |
| `afull` design matrix | principal angle 2.0e-07 rad (= IDL's float32 storage limit) |
| `decorrelatehr858` coefficients | 7.8e-11 relative |
| corrected flux (`quatcorrectonelc`) | **6.1e-13** |
| `INTERPOLATE(cubic=-0.5)` | 1.6e-13 |
| quaternion binning (vs production `.idl` save, sector 11) | 6.2e-13 relative |
| barycentric time correction | **0.00 sec** (2.3e-13 d) |

### End-to-end, short cadence

This is the cleanest end-to-end test in the repo, because it is genuinely
like-for-like: both sides start from the *same* SPOC light-curve files at the
same cadence, so cadences match one-to-one on time and any disagreement is the
pipeline alone. HIP 67522 against the author's IDL reference
(`tests/compare_hip67522_sc.py`; reference files not distributed here):

| sector | cadence | matched | corr | rms diff | p2p py/IDL | amplitude ratio |
|---|---|---|---|---|---|---|
| 11 | 120 s | 11660 | **0.99976** | 145 ppm | 647 / 628 ppm | 1.0007 |
| 38 | 120 s | 18074 | **0.99992** | 214 ppm | 663 / 654 ppm | 0.9996 |
| 64 | 20 s | 101491 | **0.99971** | 360 ppm | 1428 / 1382 ppm | 1.0004 |
| 101 | 20 s | 96808 | 0.99388 | 2382 ppm | 1551 / 1621 ppm | 1.0272 |
| 102 | 20 s | 95525 | 0.99752 | 1540 ppm | 1492 / 1562 ppm | 1.0014 |
| **all** | | 323558 | **0.99668** | 1571 ppm | | 1.0102 |

In Sectors 11, 38 and 64 the two agree to r = 0.9997+, with rms differences
(145–360 ppm) *below* the photometric noise and amplitudes matching to 0.04%.

For comparison, running the **FFI** pipeline against the same reference files
gives r = 0.933 overall — that gap is the data source, not the pipeline, and is
the reason `"auto"` prefers SPOC data where it exists.

Sectors 101 and 102 are the weaker pair. The reference keeps ~12% more cadences
there (109218 vs 97155 in Sector 101), and the extras carry quality bits 13
(stray light), 7 (cosmic ray in aperture) and 11 (Argabrightening) — none of
which `quicklooksector3.pro` admits as written, in either its default or its
`/allowscattered` branch. Forcing `allowscatteredlight=True` closes the count gap
(106161) but makes the correlation *worse* (0.990), so the mask is not the whole
story. The residual has clear rotation-phase structure and Python retains 2.7%
more amplitude in Sector 101, consistent with a slightly different systematics
fit driven by a different input cadence set. Unresolved; the older three sectors
show no such effect.

### End-to-end, FFI

Against an IDL-produced reference light curve for TIC 88785435
(not distributed with this repo). Cadence counts match
exactly in all three cadence regimes (30 min / 10 min / 200 s):

| sector | cadence | n points | correlation | p2p py/IDL | MAD-scatter py/IDL |
|---|---|---|---|---|---|
| 11 | 30 min | 1037 = 1037 | 0.9945 | 906 / 892 ppm | 17444 / 17244 ppm |
| 38 | 10 min | 3693 = 3693 | 0.918 | 1447 / 1347 ppm | 12353 / 15849 ppm |
| 65 | 200 s | 11169 = 11169 | 0.9819 | 2628 / 2374 ppm | 26180 / 26021 ppm |

Sector 65's raw correlation looks poor (0.45) only because the IDL reference
contains 101 catastrophic outliers (0.9% of points, minimum flux **−11.3**)
that this port does not produce; the numbers above exclude them.

### Known open item

**Sector 38 amplitude is ~24% low** (12353 vs 15849 ppm MAD-scatter,
correlation 0.918 at the best-matching aperture rather than 0.99). Sectors 11
and 65 agree to ~1%, and the dilution factors are consistent across all three
sectors (0.31–0.97), so this is not a dilution-model failure. Not yet resolved.
Note that this is a genuinely blended target — a brighter star sits ~2 px from
the target in every sector — so aperture choice has unusually large leverage
here, and the reference run's exact settings (cutout size, aperture overrides)
are unknown.

## Scattered light: measured effect

**It is not a looser mask. It is a different one, and it can lose you more data
than it recovers.** The IDL tests quality for *equality*, not as a bitmask:

```idl
yesscattered = a.quality eq 4096 or a.quality eq 2048
x = where(yesscattered or (a.quality eq 0 and ...))
```

Three consequences follow, none of them obvious from the keyword name:

1. **`32768` is dropped.** The default mask keeps `quality EQ 0 OR quality EQ
   32768` (bit 16, "insufficient targets for error correction" — a PDC
   complaint that says nothing about SAP). The scattered branch does not, and
   nothing puts it back.
2. **Combined flags stay excluded.** A cadence flagged `4160` (= 64 | 4096) is
   straylight *and* a cosmic ray, so it fails the equality test.
3. **Two hard-coded bad-time windows are dropped** in the FFI path but the
   third, 1530–1532, is not reinstated — that is the IDL's behaviour, kept.

Measured on real data, the flag's effect is not even consistent in sign:

| target / sector | default | with flag | change | p2p |
|---|---|---|---|---|
| TIC 299798795 s68 | 89,575 | 109,033 | **+19,458** | 5781 → 6290 ppm |
| HIP 67522 s101 | 97,155 | 106,161 | +9,006 | 1551 → 1584 ppm |
| **TIC 410214986 s68** | 107,463 | 91,687 | **−15,776** | 893 → 861 ppm |

TIC 410214986 sector 68 has 19,474 cadences flagged `32768` and only 3,233
flagged `4096`, so turning the flag on trades 19k good cadences for 3k marginal
ones. Check the quality histogram before using it.

Where it does add data, point-to-point scatter gets slightly *worse* — expected,
since these cadences are noisier. And on HIP 67522 Sector 101 it moved
correlation with the IDL reference the wrong way, 0.9939 → 0.9904, with rms
difference rising 2382 → 2927 ppm.

## 20-second CBVs

lightkurve cannot fetch these — `load_tess_cbvs` says so in its own docstring
("For now, this function will only load 2-minute cadence CBVs"). The files do
exist at MAST as `*-a_fast-cbv.fits`, sitting in the same directory as the
2-minute `*-s_cbv.fits`; they are simply absent from the bulk-download curl
script that lightkurve greps. `spoc.resolve_cbv_url` finds the 2-minute URL the
same way and swaps the suffix.

Band 3 does not always carry the eight vectors the IDL loops over — it varies by
sector and CCD (8 for HIP 67522 in Sectors 11, 38, 101 and 102; only 4 on
camera 1 / CCD 2 in Sector 64). The IDL reaches for all eight via `execute()`,
which silently returns 0 for a tag that does not exist, so it too uses only what
is there. This port takes whichever `VECTOR_k` columns the file actually has, so
the regressor count printed per sector is 12 quaternion columns plus however
many CBVs exist.

## Special sectors: 1751 in detail

Background: sector 1751 is a dedicated 3I/ATLAS campaign that interrupted
Sector 99 near the end of orbit 209, pointed at the comet field for ~7 days,
then returned to normal operations (TESS DRN **DR136**, April 2026).

**tess-point's *package* cannot see it, but its Web Tool can.** The pip
package's bundled pointing table covers sectors 1–121, so 1751 is absent — yet
the [TESS-point Web Tool](https://heasarc.gsfc.nasa.gov/wsgi-scripts/TESS/TESS-point_Web_Tool/TESS-point_Web_Tool/wtv_v2.0.py/)
returns it, because its backend carries the special-campaign pointings. The
fix is to supply the pointing yourself: `tess_stars2px` accepts a
`sectorOverrideFile` of `sector ra dec roll`, and DRN §1 publishes sector
1751's boresight as **RA 109.4916°, Dec 21.8424°, Roll 84.3356°**.

`catalog.SPECIAL_SECTOR_POINTINGS` now carries that entry and
`observed_sectors()` folds it in, so 1751 is predicted natively. Verified
against the Web Tool for TIC 47319867 — same camera 3 / CCD 1, column
1937.626 vs 1937.628, row 222.158 vs 222.155 (0.003 px). Add further special
campaigns to that dict as they appear. (For anything still outside both
tess-point and the dict, the veto is skipped and TESScut's having returned a
cutout is taken as the evidence.)

**It is past the ephemeris.** Sector 1751 runs 2026-01-15 → 01-22 (BTJD ~4056),
well beyond the bundled ephemeris (ends 2024-04-04), so barycentric times fall
back to SPOC's `TIMECORR` with a warning. Fine for transit work; if you need
sub-second absolute timing, extend the ephemeris.

**No CBVs are archived for it.** DRN §1 says only a subset of pipeline modules
were run, and lists the delivered products as calibrated TPFs for 37 comet
postage stamps, light curves for PPA targets, and calibrated 200 s FFIs — no
CBVs. (The generic list on the acknowledgements page *does* mention CBVs, but
that is boilerplate common to every DRN.) PDC itself ran, so CBVs existed
inside SPOC, they were just not delivered; nothing is retrievable from MAST,
and `tesscurl_sector_1751_cbv.sh` 404s. So `usecbv=True` is silently a no-op
and the correction runs on quaternions alone (18 vectors with `skew`/`kurt`).
Quaternions are fully available and cover all 1835 cadences.

**Caveat — two unflagged artifacts survive.** Neither is removed by the
quaternions, and CBVs (which normally handle exactly this) are unavailable:

1. **Post-safe-mode recovery.** The 2.81 d gap is *not* a routine downlink: the
   solar panel orientation was not updated for this pointing, energy generation
   was limited, and the rising battery discharge rate tripped a safe mode
   (DRN §1). The 190 cadences immediately after recovery
   (BTJD 4059.29–4059.73) ramp ~10% and are not quality-flagged. The DRN's own
   Figure 4 shows the same spike in PA-light-curve MAD.
2. **Scattered light at the end.** DRN Figure 2 shows Camera 3's background
   climbing steeply late in the sector; measured here it rises **3.8×**
   (172 → 651 e/s per pixel) over the final day. No stray-light flag (bit 12)
   is set on these FFIs, so the pipeline keeps them.

Only bits 4 (Earth Point) and 6 (RW desaturation) appear in the cutout's
quality column — 12 cadences total. Scatter by segment:

| | point-to-point | MAD scatter |
|---|---|---|
| all 1823 cadences | 5339 ppm | 11271 ppm |
| excluding both flagged regions (1345 cadences) | 5326 ppm | 8857 ppm |
| orbit A only | 4593 ppm | 5218 ppm |
| orbit B, settled | 5479 ppm | 6396 ppm |

There is also a ~2.8% median flux offset between the two orbits.
Recommendation: cut both regions and treat the orbits as separate segments
before fitting. `tests/plot_lightcurve.py` flags the high-background stretch
data-driven (background > 2× the sector median) rather than by hard-coded times.

```bash
python tests/plot_lightcurve.py 47319867 1751
```

produces `output/tic47319867_s1751.png`.
