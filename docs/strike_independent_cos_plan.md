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
5. Only after standalone tests pass, wire the same concept into PyFENG classes as an additive `price_smile(...)` path.

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

## Next Milestone

Add Junike-Pankrashkin range selection as an optional method, not as the default. Suggested API shape:

```python
setup = make_cos_smile_setup(
    char_func, texp, fwd, df,
    n_cos=256,
    trunc_range="jp",
    eps_tol=1e-10,
)
```

The exact API can change after we finish translating the paper's Markov-inequality bound into model inputs.
