# PyFENG Integration

This folder contains the COS method implementation integrated into
[PyFENG](https://github.com/PyFE/PyFENG) (Prof. Jaehyuk Choi's package).

## Files

- `sv_cos.py` — COS abstract base and BSM concrete pricer:
  - `CosABC` — abstract base (mirrors `FftABC`)
  - `CosSmileSetup` — reusable strike-independent density coefficients
    for volatility-smile pricing.
  - `BsmCos` — BSM model via COS
  - Re-exports `HestonCos` from `sv_heston_cos.py` for backward compatibility.
- `sv_heston_cos.py` — `HestonCos` class with Heston-specific σ_h
  truncation (F&O 2008 Section 5.2), per-strike Ruijter & Oosterlee (2012)
  centering, τ-adaptive `L = max(10, 3τ+2)`, and a numba JIT kernel
  with a NumPy fallback.
- `lv_cos.py` — Lévy-process COS pricers:
  - `VarGammaCos` — Variance Gamma (matches `pf.VarGammaFft` signature).
  - `CgmyCos` — CGMY infinite-activity Lévy (matches `pf.CgmyFft` signature).
- `cos_range.py` — Junike-Pankrashkin Markov truncation-range helpers.
- `bermudan_cos.py` — Bermudan call/put pricing via the F&O 2009 backward
  induction.  `BermudanCosMixin` is a model-agnostic add-on for any 1-D
  Lévy `CosABC` subclass; `BermudanBsmCos`, `BermudanVgCos`, and
  `BermudanCgmyCos` are pre-mixed concrete classes.  Heston is
  intentionally excluded -- its 2-D state requires the F&O 2009 §4
  extension that this 1-D mixin does not implement.
- `frft.py` — Carr-Madan pricing via the Bailey-Swarztrauber fractional
  FFT (Chourdakis 2005).  `FrftMixin` adds `price_frft()` to any
  `FftABC` subclass; `BsmFrft`, `VarGammaFrft`, `CgmyFrft`, and
  `HestonFrft` are pre-mixed concrete classes that decouple the
  frequency-grid spacing from the log-strike grid spacing.

## Usage with PyFENG

Copy these files into your local PyFENG package folder and add to its
`__init__.py`:

```python
from .sv_cos import BsmCos
from .sv_heston_cos import HestonCos
from .lv_cos import VarGammaCos, CgmyCos
from .bermudan_cos import BermudanBsmCos, BermudanVgCos, BermudanCgmyCos
from .frft import BsmFrft, VarGammaFrft, CgmyFrft, HestonFrft
```

Then use it like any other PyFENG model:

```python
import numpy as np
import pyfeng as pf

m = pf.HestonCos(0.0175, vov=0.5751, mr=1.5768, theta=0.0398, rho=-0.5711)
m.price(np.array([90, 100, 110]), 100, 1.0)

# Reuse one COS density setup across a whole strike smile.
setup = m.make_smile_setup(spot=100, texp=1.0)
setup.price(np.array([80, 90, 100, 110, 120]), cp=1)

# Optional Junike-Pankrashkin Markov range for models with analytic moments.
setup_jp = m.make_smile_setup(
    spot=100,
    texp=1.0,
    trunc_range="jp",
    eps_tol=1e-8,
    moment_order=8,
    payoff_bound=1.0,
)

# Bermudan put under BSM with 50 equally-spaced exercise dates (~American).
m_ber = pf.BermudanBsmCos(sigma=0.25, intr=0.10)
m_ber.n_exercise = 50
m_ber.price_bermudan(100.0, 100.0, 1.0, cp=-1)   # ~6.55
```

For one-off smile calls, use `m.price_smile(strikes, spot, texp, cp=...)`.
For repeated calls at the same model/expiry/range, prefer
`setup = m.make_smile_setup(...)` and then `setup.price(...)`; the setup
caches the grid, characteristic-function samples, phase shift, and
prime-weighted density coefficients.  The payoff coefficients remain
strike-dependent and are rebuilt vectorially.

`trunc_range="jp"` requests the Junike-Pankrashkin (2022) Markov range.
BSM, Variance Gamma, and CGMY provide analytic cumulants through order 8.
Heston keeps the F&O sigma-h range by default; a JP-style Heston 8th
moment estimator is a separate follow-up.

`numba` is optional. If installed, `HestonCos.price` runs the JIT kernel
on first call (pay the cost upfront with `from pyfeng.sv_heston_cos import
warmup_numba; warmup_numba()`). Without numba, the same code runs
interpreted — slower but functionally identical.
