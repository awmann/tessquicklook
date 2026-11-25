"""Offline checks for tessquicklook.scattered (sl_flag bits).  Run: python tests/test_scattered.py"""
import numpy as np
from tessquicklook import scattered as S

fails = 0
def check(name, cond):
    global fails
    print(("ok   " if cond else "FAIL ") + name); fails += (not cond)

n = 300
t = np.arange(n) / 48.0
q = np.zeros(n, int)
sky = np.full(n, 100.0); sky[100:110] = 250.0; sky[200:205] = 140.0   # e=1.5 and e=0.4
cube = np.tile(sky[:, None, None], (1, 15, 15)) + np.random.default_rng(1).normal(0, 0.5, (n, 15, 15))
cube[:, :, :5] = 0.0                                                  # zero-padded edge columns

s, e, f = S.ffi_flags(cube, q, t)
check("padded pixels ignored: quiet excess ~0", abs(np.median(e)) < 0.02)
check("bit1 on the e=1.5 block, dilated by one cadence", np.array_equal(np.flatnonzero(f & 1), np.arange(99, 111)))
check("bit8 covers bit1 and the e=0.4 block", set(np.flatnonzero(f & 8)) == set(range(99, 111)) | set(range(199, 206)))
check("no bit128 when sky exists", not (f & 128).any())

off = np.random.default_rng(2).normal(0, 0.05, (n, 15, 15))           # stamp off the science pixels
s, e, f = S.ffi_flags(off, q, t)
check("off-silicon stamp -> all bit128, excess NaN", (f == 128).all() and np.isnan(e).all())

pad = np.zeros((n, 15, 15)); pad[:, :3, :3] = 100.0                    # <20 real pixels
check("too few real pixels -> bit128", (S.ffi_flags(pad, q, t)[2] == 128).all())

shuf = np.random.default_rng(3).permutation(n)                        # dilation follows time, not row order
_, _, fs = S.ffi_flags(cube[shuf], q[shuf], t[shuf])
check("dilation in time order for unsorted input", np.array_equal(fs, S.ffi_flags(cube, q, t)[2][shuf]))

qs = np.zeros(n, int); qs[10] = 32768; qs[20] = 4096; qs[30] = 32768 | 4096
bk = np.full(n, 50.0); bk[150:160] = 120.0
e, f = S.spoc_flags(bk, qs)
check("2-min bit2 on 32768", set(np.flatnonzero(f & 2)) == {10, 30})
check("2-min bit4 on 4096", set(np.flatnonzero(f & 4)) == {20, 30})
check("2-min bit1 from SAP_BKG, no dilation", set(np.flatnonzero(f & 1)) == set(range(150, 160)))
check("dtypes", f.dtype == np.uint8 and e.dtype == np.float32)

raise SystemExit(1 if fails else print("all passed"))
