# IDL reference light curves

Output of the original IDL pipeline (`quicklooksector3.pro` /
`quicklooktessffi.pro`, the ones this repo ports), for cross-checking the
Python port. Columns: `time` (BTJD), `flux` (normalized), `cad` (cadence,
seconds).

| file | TIC | star |
|---|---|---|
| `27491137_idl.csv` | 27491137 | TOI-2076 |
| `257605131_idl.csv` | 257605131 | TOI-451 |

See `toi2076_e_validation/idl_vs_python.py` for the comparison against the
Python output.
