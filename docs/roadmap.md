# Post-Presentation Extension Status

This document records the roadmap and current implementation status for the two post-presentation extensions: reusable strike-independent COS setup and Black-Scholes control variates. Both are implemented as opt-in additions; the baseline Fang-Oosterlee `price(...)` path remains unchanged.

## Implemented

- **Reusable strike-independent setup in `cos_pricing`.** `CosSmileSetup`, `make_cos_smile_setup(...)`, and `cos_price_smile(...)` cache the density-side objects for a fixed model, expiry, forward, discount factor, and range. Strike-dependent payoff coefficients are still built vectorially, but characteristic-function samples and prime-weighted density coefficients are reused.
- **PyFENG smile API.** `make_smile_setup(...)` and `price_smile(...)` are available on the PyFENG COS classes for BSM, Heston, VG, and CGMY. These methods are additive; existing `price(...)` calls keep the old behavior.
- **Junike-Pankrashkin Markov range helpers.** BSM, VG, and CGMY expose analytic cumulants through order 8 and can request `trunc_range="jp"` / `jp_trunc_range(...)`. Heston keeps the Fang-Oosterlee sigma-h range by default until a dedicated high-order Heston moment estimator is added.
- **Black-Scholes control variates.** Heston, VG, and CGMY expose opt-in `price_cv(...)` style methods. The simple method uses average variance for Heston and variance matching for VG/CGMY. The Joshi-Yang real-axis and half-contour volatility selectors are also implemented as explicit `vol_method` choices.
- **Validation coverage.** Automated tests cover default-path preservation, smile setup equivalence, JP finite ranges, control-variate correction identities, PyFENG source consistency, and fixed-seed randomized robustness across BSM, Heston, VG, and CGMY.

## Why The Extensions Matter

The practical target is volatility-smile pricing: many strikes at the same maturity. The expensive density side of COS does not depend on strike, while the payoff boundary does. A reusable setup avoids rebuilding the density-side grid and CF samples for every smile call, and the control variate can reduce coarse-grid truncation error when the COS resolution is intentionally small.

## Strike-Independent COS Setup

For a fixed model and expiry, the reusable objects are:

- cosine grid `u_k = k*pi/(b-a)`
- characteristic-function samples `phi(u_k)`
- phase-shifted, prime-weighted real density coefficients
- forward and discount factor

The strike-dependent part is the payoff boundary `log(K/F)`, so payoff coefficients still depend on strike. The implementation therefore reuses the density side and vectorizes the payoff side over all strikes.

Main APIs:

```python
from cos_pricing import make_cos_smile_setup

setup = make_cos_smile_setup(cf, texp, fwd, df, n_cos=256, trunc_range=(a, b))
prices = setup.price(strikes, cp=1)
```

PyFENG-style usage:

```python
setup = model.make_smile_setup(spot=100.0, texp=1.0)
prices = setup.price(strikes, cp=1)

setup_jp = model.make_smile_setup(
    spot=100.0,
    texp=1.0,
    trunc_range="jp",
    eps_tol=1e-8,
    moment_order=8,
)
```

## Junike-Pankrashkin Range Status

The current JP implementation is the Markov-bound milestone:

- `jp_markov_range(...)` converts an even central moment into a truncation interval.
- BSM, VG, and CGMY provide analytic cumulants through order 8.
- The JP path is not the default; callers must request it explicitly.

Remaining JP work:

- Add a Heston-specific high-order moment estimator. Junike-Pankrashkin Section 4.4 suggests approximating the 8th moment upfront rather than differentiating the Heston characteristic function on every calibration call.
- Compare JP ranges with Le Floc'h-style ranges on the same parameter grid and document when each is tighter.
- Coordinate with Nigel's team before changing any defaults.

## Black-Scholes Control Variate Status

The implemented correction is:

```text
model_COS + (BS_exact - BS_COS)
```

The Black-Scholes COS leg uses the same COS-style range as the target model whenever possible, so the correction targets the same truncation/series error.

Available volatility choices:

- `vol_method="simple"`: average variance for Heston; variance matching for VG/CGMY.
- `vol_method="joshi"`: Joshi-Yang real-axis derivative match.
- `vol_method="joshi-half"`: Joshi-Yang eta=1/2 contour match.

The notebook shows the expected behavior: control variates help most at coarse `N`; once COS has converged, the correction term is close to zero.

## Remaining Work

- Add Heston JP high-order moment support.
- Add a larger smile benchmark grid that reports speedup versus strike count for scalar loops, vectorized `price(...)`, `price_smile(...)`, and `price_smile_cv(...)`.
- Keep CGMY `Y=1.98` documented as a numerical-convention/reference caveat rather than presenting it as a clean reproduction.
- If contributing upstream to PyFENG, keep all new APIs opt-in and discuss defaults with the maintainers first.
