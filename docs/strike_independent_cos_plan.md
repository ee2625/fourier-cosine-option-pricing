# Strike-Independent COS Smile Plan

Goal: price a whole volatility smile with one COS density setup instead of rebuilding strike-dependent objects for every strike.

## Safe Rollout

1. Keep existing `price(...)` and `cos_price(...)` behavior unchanged.
2. Add an explicit reusable setup path:
   - `make_cos_smile_setup(...)`
   - `CosSmileSetup.price(...)`
   - `cos_price_smile(...)`
3. Validate the setup path against the existing COS engine using the same fixed truncation range.
4. Only after that, add improved truncation-range selection such as Junike-Pankrashkin (2022).
5. Wire the same concept into PyFENG classes as an additive `price_smile(...)` path.

## What Is Reused

For a fixed model, expiry, forward, discount factor, and interval `[a, b]`, the following are strike-independent:

- cosine grid `u_k = k*pi/(b-a)`
- characteristic-function samples `phi(u_k)`
- phase-shifted, prime-weighted density coefficients

The strike-dependent part is the payoff boundary `log(K/F)`, so payoff coefficients are still vectorized over strikes. The expensive density side is cached once.

## First Milestone

Implemented first with an explicit or legacy truncation range only. This intentionally separates two problems:

- reusable strike-independent COS setup
- Junike-Pankrashkin error-controlled range selection

The first milestone is correct when `cos_price_smile(...)` matches `cos_price(...)` for the same `[a, b]` and `N`.

## PyFENG Milestone

Implemented as an additive interface on the PyFENG COS classes:

- `make_smile_setup(spot, texp, trunc_range=None)`
- `CosSmileSetup.price(strike, cp=...)`
- `price_smile(strike, spot, texp, cp=..., trunc_range=None)`

The legacy `price(...)` methods are unchanged.  For BSM, VG, and CGMY,
the setup uses each model's existing strike-independent truncation range
unless an explicit range is supplied.  For Heston, the setup uses the same
F&O half-width as the Numba-backed `price(...)`, but translated into the
strike-independent log-forward variable `log(S_T/F)`.

## Junike-Pankrashkin Milestone

Implemented the first optional JP range selector:

- `jp_markov_range(...)` converts a central moment into the Corollary 9
  Markov interval.
- `BsmModel.jp_trunc_range(...)`, `VgModel.jp_trunc_range(...)`, and
  `CgmyModel.jp_trunc_range(...)` provide exact model cumulants through
  the requested even moment order.
- PyFENG `CosABC.make_smile_setup(..., trunc_range="jp", eps_tol=...,
  moment_order=..., payoff_bound=...)` can request the same range when the
  model supplies JP cumulants.  BSM, VG, and CGMY provide analytic
  cumulants through order 8.

This is still not the default.  The existing F&O ranges remain unchanged
unless the caller explicitly asks for `trunc_range="jp"` or calls
`jp_trunc_range(...)`.

## Remaining Milestone

Add a Heston-specific high-order moment estimator.  JP Section 4.4 suggests
approximating the 8th moment upfront, rather than differentiating the
Heston characteristic function on every calibration call.  Until we add
that estimator, Heston keeps its existing F&O sigma-h smile range by
default.  Generic JP calls on Heston should use a lower supported moment
order or a future Heston-specific moment approximation.

Suggested API shape:

```python
setup = model.make_smile_setup(
    spot=100.0,
    texp=1.0,
    trunc_range="jp",
    eps_tol=1e-10,
    moment_order=8,
    payoff_bound=1.0,
)
prices = setup.price(strikes, cp=1)
```
