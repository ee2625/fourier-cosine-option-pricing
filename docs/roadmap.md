# Roadmap — post-presentation extensions

Two extensions queued up for after the Thursday presentation. Both target the same practical use case: pricing a **whole volatility smile** (many strikes at the same maturity) faster and with tighter error control than the current engine.

## Context: why this matters

The current `cos_price` engine already shares the heavy work — characteristic-function evaluation, truncation range, the `[a, b]` interval — across strikes. The dominant per-call cost is one $(M \times N)$ matrix-vector product, which is $M$ strikes for the cost of one. So smile pricing is already pretty fast.

Where it can be sharper:

1. **The payoff coefficient $V_k$ still depends on $K$.** Look at [cos_method.py:90](../src/cos_pricing/cos_method.py#L90) — `log_kk = np.log(strike / fwd)` enters the trig phase $u_k \cdot \log(K/F)$, so the `(M, N)` matrix `W` is built per strike. With Junike-Pankrashkin (2022) and Le Floc'h (2020) we can factor this differently and do the heavy work once.

2. **Truncation-range error is bounded only heuristically.** The current rule (paper Eq. 49: $b - a = L \sqrt{|c_2| + \sqrt{|c_4|}}$ with $L = 10$ as a safety multiplier) is not tied to a target accuracy. Junike-Pankrashkin give a sharp upper bound on the COS truncation error → you can set $[a, b]$ to hit a target tolerance directly.

3. **No control variate.** The COS price has the form (true price) + (truncation error) + (series-truncation error). For the Black-Scholes case, the *exact* price is known. We can use that as a control variate against the COS price for any model whose Black-Scholes "equivalent vol" is computable. Joshi & Yang (2011) did this for FFT pricing. Doing it for COS with $\sigma_{\text{eq}} = \sqrt{E[\bar V_T]}$ (closed form for Heston) would be the new contribution.

---

## Extension A — Strike-independent COS engine (Junike-Pankrashkin 2022 + Le Floc'h 2020)

### Reading list

- **Junike, G. and Pankrashkin, K. (2022)** *Precise option pricing by the COS method — How to choose the truncation range.* Applied Mathematics and Computation 421:126935. <https://doi.org/10.1016/j.amc.2022.126935>
- **Le Floc'h, F. (2020)** *More robust pricing of European options based on Fourier cosine series expansions.* SSRN. PDF in repo: [`More Robust Pricing of European Options Based on Fourier Cosine Series Expansions.pdf`](../More%20Robust%20Pricing%20of%20European%20Options%20Based%20on%20Fourier%20Cosine%20Series%20Expansions.pdf).
- **Reference for current engine**: [`src/cos_pricing/cos_method.py`](../src/cos_pricing/cos_method.py).

### What changes

**Math.** Switch the cosine basis from $x = \log(S_T / F)$ (log-moneyness, mildly K-dependent via $F$) to $y = \log(S_T / S_0)$ (log-return, K-independent). Then:

- Density coefficients $A_k = (2/(b - a)) \mathrm{Re}[\varphi_Y(u_k) \exp(-i u_k a)]$ are computed **once** for the whole smile.
- Payoff coefficients $V_k(K) = \int_{\log(K/S_0)}^{b} (S_0 e^y - K) \cos(u_k (y - a))\, dy$ — the integration limit moves with $K$, but the integrand is structured so the closed form factors as
  $$V_k(K) \;=\; S_0 \cdot \tilde\chi_k(\log(K/S_0), b) \;-\; K \cdot \tilde\psi_k(\log(K/S_0), b),$$
  where $\tilde\chi_k$ and $\tilde\psi_k$ are closed-form integrals (chi/psi family), and the K-dependence enters only through $\sin(u_k \log(K/S_0))$ and $\cos(u_k \log(K/S_0))$ — one trig pair per $(K, k)$.
- Vectorise: build the $(M, N)$ matrices of trig values in one shot (`np.outer`-style), then a single matvec gives all $M$ prices.

**Truncation range.** Replace the cumulant heuristic with the Junike-Pankrashkin sharp bound: choose $[a, b]$ such that the truncation error is $\le \epsilon_{\text{tol}}$ (user-specified), via their explicit error formula. Le Floc'h's recipe is similar — both give $L$ as a function of $(\epsilon_{\text{tol}}, \text{model parameters})$ instead of a fixed safety multiplier.

### API sketch

Two pieces, both backwards-compatible:

```python
# 1) New entry point with explicit error control:
cos_price_smile(
    char_func, texp, strikes, spot, intr=0.0, divr=0.0,
    cp=1, n_cos=128, eps_tol=1e-10,                # NEW: target accuracy
    cumulants=None,                                # OPTIONAL: pass exact cumulants
)

# 2) Existing cos_price() unchanged (legacy callers + tests don't break)
```

Internally `cos_price_smile` computes:
- Strike-independent: `cf_vals = char_func(u_arr)`, the centering phase, and the density coefficients `A_k`.
- Strike-dependent (vectorised across all `M` strikes): `V_k(K)` via two trig matrices (one for $\sin$, one for $\cos$) plus the closed-form $\tilde\chi$ / $\tilde\psi$ formulas.
- Final dot product: `prices = df * (V @ A)` — same `(M, N)` matvec cost as the existing engine but with a strike-independent `A`.

### Implementation order

1. Read both papers carefully (start with Le Floc'h since the PDF is local).
2. Derive the new $V_k(K)$ closed form on paper (one-page derivation).
3. Implement `cos_price_smile()` alongside `cos_price()` — keep both working. Validate `cos_price_smile` reproduces every existing benchmark table to the same accuracy.
4. Add Junike-Pankrashkin error-controlled $[a, b]$ as an optional path (`eps_tol=...` keyword).
5. Write a benchmark example: time-vs-strike-count comparison of `cos_price` (M scaled cost) vs `cos_price_smile` (M ~ amortised cost).
6. Coordinate with Nigel's team — they may already have partial work here.

### Validation plan

- Every existing table (Tables 1, 2, 3, 4-6, 7, 8-10, Bermudan) must reproduce to current accuracy with the new engine.
- Add a new benchmark in [examples/](../examples/): time per smile (e.g. 21 strikes from $\{50, 55, \ldots, 150\}$) under Heston, compared against the current engine.
- Tests: replicate the existing strike-dependent tests but call `cos_price_smile` instead. Should be drop-in.

---

## Extension B — Black-Scholes control variate

### Reading list

- **Joshi, M. S. and Yang, C. (2011)** *Fourier Transforms, Option Pricing and Controls.* SSRN Working Paper. <https://ssrn.com/abstract=1941464>
- **Ball & Roma (1994)** for the closed-form Heston average variance: $E[\bar V_T] = \bar v + (v_0 - \bar v)(1 - e^{-\kappa T})/(\kappa T)$.

### What changes

**Math.** For any model with CF $\varphi$, define
$$C_{\text{model}}(K) \;\approx\; \underbrace{C_{\text{model,COS}}(K)}_{\text{biased, fast}} \;+\; \underbrace{\big[\, C_{\text{BS,exact}}(K, \sigma_{\text{eq}}) - C_{\text{BS,COS}}(K, \sigma_{\text{eq}}) \,\big]}_{\text{control: truncation/series-truncation noise of the BS run, computed in closed form against the exact BS price}}$$

The bracketed correction is the COS error for the BS model at the equivalent vol $\sigma_{\text{eq}}$. If this error is highly correlated with the COS error for the *target* model — which it is when both share the same $[a, b]$ truncation and $N$ series cutoff — the correction cancels most of the error, sharpening the answer for free.

**Choice of $\sigma_{\text{eq}}$ — the user's suggestion: $\sigma_{\text{eq}} = \sqrt{E[\bar V_T]}$**, where $\bar V_T = (1/T) \int_0^T v_t\, dt$ is the average instantaneous variance.

For Heston this has a closed form (Ball-Roma 1994):
$$E[\bar V_T] \;=\; \bar v \;+\; (v_0 - \bar v) \cdot \frac{1 - e^{-\kappa T}}{\kappa T}.$$

PyFENG's `HestonABC.avgvar_mv(texp)` returns the mean and variance of $\bar V_T$ — the first element is what we need.

For VG, the analogous quantity is just the constant $\sigma^2 + \nu \theta^2$ (the second cumulant per unit time of the VG log-return process). For CGMY, it's $C \Gamma(2 - Y)(M^{Y-2} + G^{Y-2})$ (the variance per unit time of the CGMY process). Both are closed-form.

**Why this is novel.** Joshi-Yang (2011) did the analogous thing for **Carr-Madan FFT pricing**. They use a more complicated $\sigma_{\text{eq}}$ choice that requires solving a non-linear equation. The PyFENG-based $\sigma_{\text{eq}} = \sqrt{E[\bar V_T]}$ is much simpler and is **strike-independent**, so it composes cleanly with Extension A — compute $\sigma_{\text{eq}}$ once, run two COS calls (target model + BS at $\sigma_{\text{eq}}$), apply the correction. Doing this for COS specifically is the publishable angle.

### API sketch

```python
# Add an optional control_variate flag to the model price methods:
m = HestonCOSPricer(S0=100, v0=0.04, ...)
m.price_call(K=100.0, tau=1.0, N=64, control_variate=True)
#                                    ^^^^^^^^^^^^^^^^^^^^
# When True:
#   1. Compute sigma_eq = sqrt(avgvar_mv(tau)[0])
#   2. Compute C_BS_exact(K, sigma_eq) via closed-form BSM
#   3. Compute C_BS_COS(K, sigma_eq) via the same N, L, [a, b] as the Heston run
#   4. Return: C_Heston_COS + (C_BS_exact - C_BS_COS)

# Free-function variant:
price_call_heston_cv(S0, K, tau, ..., N, L)   # always uses the control variate
```

### Validation plan

- For Heston: at small $N$ (where COS error is non-negligible), the control variate should give an error reduction of 1-3 orders of magnitude. Plot error vs $N$ for both the plain and CV versions.
- For VG / CGMY: similar.
- Sanity: when the model **is** BSM, $\sigma_{\text{eq}} = \sigma$ exactly, and the CV formula collapses to $C_{\text{exact}}$ trivially — confirms no bias is introduced.

### Implementation order

1. Implement first as a wrapper: `def price_with_cv(model, K, T, ...): ...` — calls the existing pricer twice and applies the correction.
2. Validate error reduction on Heston / VG / CGMY at small $N$. Plot error-vs-N for plain vs CV.
3. If the gain is meaningful, integrate into the model classes as a `control_variate=True` keyword.

---

## Implementation sequencing

Both extensions are independent. The cleanest order:

1. **Extension B first (control variate)** — smaller, isolated, doesn't refactor existing code. Each model class gains an extra keyword. Can be merged without disrupting the Junike work in flight by Nigel's team.
2. **Extension A second (strike-independent engine)** — bigger refactor. The new engine should support the control variate as well, so doing B first means the API is settled.

Together, the engine becomes:
- $M$ strikes priced for one matvec
- One CF evaluation per smile
- One BS-equivalent vol per smile, and one BS-COS run per smile, gives a control-variate corrected price for *every strike*
- Truncation range tied to a target tolerance, not a heuristic

This is the smile-pricing performance / accuracy frontier for any single-factor exponential-Lévy model, plus Heston via PyFENG's Heston-CGMY composition. Strong material for either an extended report or a follow-up paper.

---

## Open questions for Erce / Nigel's team / prof

1. **Junike-Pankrashkin vs Le Floc'h.** Both offer strike-independent ranges; do we adopt one or compose them? Le Floc'h's recipe is more practical (cumulant-based with explicit error term), Junike-Pankrashkin's bound is sharper but harder to evaluate. Erce's prof's note says J-P is "more advanced" — does that mean we go with J-P only, or use Le Floc'h as a sanity check?

2. **For VG/CGMY: what's the right BS-equivalent vol?** $\sigma_{\text{eq}} = \sqrt{E[\bar V_T]}$ is natural for Heston (variance is a state). For pure-jump Lévy (VG, CGMY), the analogous quantity is the variance per unit time, but the "BS-like" payoff dynamics differ. May need to test empirically.

3. **Heston Bermudan via 2D COS (Ruijter-Oosterlee 2012).** Adjacent to Extension A. Once the strike-independent engine is in place, the 2D Bermudan extension to Heston is a natural follow-on. Out of scope for these two extensions but worth flagging.

4. **Coordinate with Nigel's team.** What's their current state on Junike implementation? May save us duplication.
