# PyFENG Integration

This folder contains the COS method implementation integrated into
[PyFENG](https://github.com/PyFE/PyFENG) (Prof. Jaehyuk Choi's package).

## Files

- `sv_cos.py` — COS abstract base and BSM concrete pricer:
  - `CosABC` — abstract base (mirrors `FftABC`)
  - `BsmCos` — BSM model via COS
  - Re-exports `HestonCos` from `sv_heston_cos.py` for backward compatibility.
- `sv_heston_cos.py` — `HestonCos` class with Heston-specific σ_h
  truncation (F&O 2008 Section 5.2), per-strike Ruijter & Oosterlee (2012)
  centering, τ-adaptive `L = max(10, 3τ+2)`, and a numba JIT kernel
  with a NumPy fallback.

## Usage with PyFENG

Copy both files into your local PyFENG package folder and add to its
`__init__.py`:

```python
from .sv_cos import BsmCos
from .sv_heston_cos import HestonCos
```

Then use it like any other PyFENG model:

```python
import numpy as np
import pyfeng as pf

m = pf.HestonCos(0.0175, vov=0.5751, mr=1.5768, theta=0.0398, rho=-0.5711)
m.price(np.array([90, 100, 110]), 100, 1.0)
```

`numba` is optional. If installed, `HestonCos.price` runs the JIT kernel
on first call (pay the cost upfront with `from pyfeng.sv_heston_cos import
warmup_numba; warmup_numba()`). Without numba, the same code runs
interpreted — slower but functionally identical.
