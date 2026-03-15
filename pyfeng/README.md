# PyFENG Integration

This folder contains the COS method implementation integrated into
[PyFENG](https://github.com/PyFE/PyFENG) (Prof. Jaehyuk Choi's package).

## Files

- `sv_cos.py` — drop-in following PyFENG class hierarchy:
  - `CosABC` — abstract base (mirrors `FftABC`)
  - `BsmCos` — BSM model via COS
  - `HestonCos` — Heston model via COS

## Usage with PyFENG

Copy `sv_cos.py` into your local PyFENG package folder and add to `__init__.py`:
```python
from .sv_cos import BsmCos, HestonCos
```

Then use it like any other PyFENG model:
```python
import numpy as np
import pyfeng as pf

m = pf.HestonCos(0.04, vov=0.5, mr=1.5, rho=-0.7)
m.price(np.array([90, 100, 110]), 100, 1.0)
```
