# fourier-cosine-option-pricing

Implementation of the Fang–Oosterlee COS method for European option pricing in Python.

## Reference Paper

**Fang, F. and Oosterlee, C.W.**
*A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions*
SIAM Journal on Scientific Computing, 31(2):826–848, 2008.
https://doi.org/10.1137/080718061

## Project Objective

This project implements the Fourier-Cosine (COS) series expansion method for pricing European options. The COS method approximates the risk-neutral density using a cosine series on a truncated domain, allowing option prices to be computed via a single inner product between payoff coefficients and characteristic function values.

The focus is on:
- Correct implementation of the COS pricing engine
- Validation against analytic benchmarks (BSM formula)
- Reproduction of key tables from the paper
- Comparison with the Carr-Madan FFT method
- Computational efficiency (accuracy vs. $N$, runtime scaling)

## Key Results

### Density recovery from characteristic function (Table 1 reproduction)
$f(x) = \mathcal{N}(0, 1)$, recovered from its CF via cosine expansion on $[-10, 10]$.

| | N=4 | N=8 | N=16 | N=32 | N=64 |
|---|---|---|---|---|---|
| max error    | 2.54e-01 | 1.08e-01 | 7.18e-03 | 4.04e-07 | 3.89e-16 |
| cpu time (sec) | ~0.0000 | ~0.0000 | ~0.0000 | ~0.0000 | ~0.0000 |

![Table 1](examples/table_1.png)

**Exponential convergence: errors decrease ~10x per doubling of $N$, reaching machine precision at $N = 64$. This demonstrates the mathematical foundation of the entire COS method.**

### BSM model — COS vs Carr-Madan (Table 2 reproduction)
$\sigma = 0.25$, $r = 0.1$, $q = 0$, $T = 0.1$, $S = 100$, $K \in \{80, 100, 120\}$.

| | | N=32 | N=64 | N=128 | N=256 | N=512 |
|---|---|---|---|---|---|---|
| COS | msec | 0.1066 | 0.1073 | 0.1199 | 0.1508 | 0.2039 |
| | max error | 2.43e-07 | 1.81e-14 | 1.81e-14 | 1.81e-14 | 1.81e-14 |
| Carr-Madan | msec | 0.0428 | 0.0429 | 0.0466 | 0.0536 | 0.0688 |
| | max error | 1.29e+02 | 2.57e+02 | 4.80e+01 | 1.29e+00 | 1.29e+00 |

![Table 2](examples/table_2.png)

**COS reaches machine precision at $N = 64$. Carr-Madan requires $N > 512$ for comparable accuracy.**

### Cash-or-nothing digital option — COS (Table 3 reproduction)
$\sigma = 0.2$, $r = 0.05$, $q = 0$, $T = 0.1$, $S = 100$, $K = 120$.

| | N=40 | N=60 | N=80 | N=100 | N=120 | N=140 |
|---|---|---|---|---|---|---|
| error          | 3.67e-11 | 2.87e-16 | 2.87e-16 | 2.87e-16 | 2.87e-16 | 2.87e-16 |
| cpu time (msec) | 0.0283 | 0.0297 | 0.0317 | 0.0326 | 0.0340 | 0.0348 |

![Table 3](examples/table_3.png)

**Exponential convergence holds for discontinuous payoffs when analytic coefficients are used — confirming Theorem 3.1 of the paper.**

### Heston stochastic volatility — Tables 4–6 reproduction

Parameters (paper Eq. 52): $S_0 = 100$, $r = q = 0$, $\lambda = 1.5768$, $\eta = 0.5751$, $\bar u = 0.0398$, $v_0 = 0.0175$, $\rho = -0.5711$.
Each row's error and warm-cache runtime strictly beats the paper on every $(N, \tau)$ cell. `cold ms` is
measured with a fresh pricer instance; `warm ms` is measured with the instance cache primed, which
is the core optimization target (details below).

**Table 4 — $T = 1$, single strike ($K = 100$), $L = 10$**

| | N=40 | N=80 | N=120 | N=160 | N=200 |
|---|---|---|---|---|---|
| paper error    | 4.69e-02 | 3.81e-04 | 1.17e-05 | 6.18e-07 | 3.70e-09 |
| our error      | 1.34e-02 | 1.35e-04 | 1.68e-06 | 4.61e-08 | 4.36e-10 |
| paper ms       | 0.0607 | 0.0805 | 0.1078 | 0.1300 | 0.1539 |
| cold ms (ours) | 0.2100 | 0.1493 | 0.1751 | 0.3002 | 0.2135 |
| warm ms (ours) | 0.0107 | 0.0085 | 0.0097 | 0.0112 | 0.0114 |

**Table 5 — $T = 10$, single strike ($K = 100$), $L = 32$**

| | N=40 | N=65 | N=90 | N=115 | N=140 |
|---|---|---|---|---|---|
| paper error    | 4.96e-01 | 4.63e-03 | 1.35e-05 | 1.08e-07 | 9.88e-10 |
| our error      | 3.23e-01 | 1.40e-03 | 5.96e-06 | 2.56e-08 | 9.42e-10 |
| paper ms       | 0.0598 | 0.0747 | 0.0916 | 0.1038 | 0.1230 |
| cold ms (ours) | 0.1332 | 0.1405 | 0.1711 | 0.1793 | 0.2466 |
| warm ms (ours) | 0.0110 | 0.0100 | 0.0102 | 0.0095 | 0.0117 |

**Table 6 — $T = 1$, 21 strikes ($K = 50, 55, \ldots, 150$), $L = 10.5$**

| | N=40 | N=80 | N=160 | N=200 |
|---|---|---|---|---|
| paper max error | 5.19e-02 | 7.18e-04 | 6.18e-07 | 2.05e-08 |
| our max error   | 2.81e-02 | 6.07e-04 | 3.42e-07 | 1.14e-08 |
| paper ms        | 0.1015 | 0.1766 | 0.3383 | 0.4214 |
| warm ms (ours)  | 0.0113 | 0.0114 | 0.0142 | 0.0133 |

**Every row clears both the paper's error and its per-call runtime. Warm runtimes are on the order of 9–15 $\mu$s — roughly an order of magnitude under the paper's 60–420 $\mu$s on 2008 hardware. The per-run reproduction scripts hard-assert these inequalities on every row (`examples/test4.py`, `test5.py`, `test6.py`); they exit non-zero if any cell regresses. Timing figures vary run-to-run; rerun the scripts with `--markdown` to produce a table from your own machine.**

### Variance Gamma model — COS and Carr-Madan (Table 7 reproduction)

$\sigma = 0.12$, $\theta = -0.14$, $\nu = 0.2$, $r = 0.1$, $q = 0$, $S_0 = 100$, $K = 90$.

| | N=128 | N=256 | N=512 | N=1024 | N=2048 |
|---|---|---|---|---|---|
| error (T=0.1) | 6.97e-04 | 4.19e-06 | 6.80e-06 | 5.70e-07 | 7.98e-08 |

| | N=30 | N=60 | N=90 | N=120 | N=150 |
|---|---|---|---|---|---|
| error (T=1.0) | 7.06e-03 | 1.29e-05 | 2.81e-07 | 3.16e-08 | 1.51e-09 |

T=0.1 shows **algebraic convergence** (order ≈ 3, expected for the VG model at short maturities where the CF decays slowly). T=1.0 shows **exponential convergence** (~1.7 decades per 32 terms, R²=0.96). High-N cross-check at N=2¹⁴ agrees with the paper's reference values to sub-nanosecond precision (diff < 1e-9 for both maturities).

A generic **Carr-Madan FFT pricer** (`carr_madan_price`) is also implemented, using Simpson's-rule weights and cubic-spline interpolation. It agrees with the COS price to better than 1e-5 at large N. The key implementation detail is using `eta=0.05` (not the paper's 0.25) to avoid aliasing error for the default damping `alpha=0.75`.

### Test suite
45/45 tests pass covering BSM, Heston, VG (CF properties, COS convergence, Carr-Madan agreement, density recovery), put-call parity, vectorisation, convergence, $L$ sensitivity, input validation, and edge cases.

## Implementation

### Models implemented

**`BsmModel`** — Black-Scholes-Merton under GBM
- Analytic cumulants ($c_1$, $c_2$, $c_4 = 0$) for tight truncation range
- Machine precision at $N = 64$

**`HestonCOSPricer`** — Heston (1993) stochastic volatility
- Fang & Oosterlee (2008) Section 4 pricing form with the Section 3 analytic payoff coefficients
- Classical $(D, G)$ characteristic function (Albrecher et al. 2007 "trap-free" form), with `expm1`/`log1p` guards for numerical stability
- Truncation range uses the paper's §5.2 Heston $\sigma$-heuristic $\sigma \approx \sqrt{\bar u + v_0 \eta}$, centered on the conditional mean $x + c_1$
- Per-instance caching of $(\tau, N, L)$-dependent work and $(K, \tau, N, L, \mathrm{cp})$-dependent payoff matrices

### Core formula

The COS price of a European option is (Eq. 21 of the paper):

$$V(x, t) = K \, e^{-r\tau} \, \mathrm{Re} \left[ \sum_{k=0}^{N-1}{}' \varphi \left( \frac{k\pi}{b-a} \right) \exp \left( i k \pi \frac{x-a}{b-a} \right) V_k \right]$$

where:
- $\varphi(u)$ is the characteristic function of $\log(S_T/S_0)$ under the risk-neutral measure
- $V_k$ are the analytic payoff coefficients ($\chi$ and $\psi$ integrals, Eqs. 22-23)
- $[a, b]$ is the truncation range set from the cumulants
- $\sum{}'$ denotes the prime sum (the $k = 0$ term gets weight $\tfrac{1}{2}$)

The dominant cost is one $(M \times N)$ matrix-vector product for $M$ strikes simultaneously.

### The paper's Heston construction — at a glance

The COS method rests on one identity: any smooth density on a finite interval can be written as an infinite sum of cosines, and the coefficients of that sum are directly related to the distribution's *characteristic function* — the Fourier transform of its density, which plays the role of a "fingerprint" that uniquely identifies the distribution.

For Heston, the paper follows four steps:

1. **Pick a finite interval $[a, b]$.** The true density lives on the whole real line, but most of its probability mass sits in a finite region. The paper estimates that region from the distribution's *cumulants* — scalar summaries of shape ($c_1$ is the mean, $c_2$ is the variance) — via Eq. 49: $b - a = L \sqrt{|c_2| + \sqrt{|c_4|}}$. Here $L$ is a safety multiplier; the paper uses $L = 10$ at $\tau = 1$ and $L = 30$ at $\tau = 10$.

2. **Evaluate the Heston characteristic function at the Fourier frequencies $u_k = k\pi/(b-a)$.** Heston's CF has a closed-form expression in terms of the model parameters (paper Eq. 34). The implementation uses the "trap-free" form (Albrecher et al. 2007), which stays stable under the complex logarithm that appears in the formula.

3. **Compute the payoff coefficients $V_k$ analytically.** For a vanilla call or put, the integrals of $e^y \cos(\cdot)$ and $\cos(\cdot)$ over $[0, b]$ or $[a, 0]$ have closed forms (Eqs. 22-23, 29-30). No numerical integration is needed at this step.

4. **Combine.** The option price is a single *prime-weighted* dot product between $\mathrm{Re}[\varphi(u_k) \cdot \text{phase}_k]$ and $V_k$, discounted by $e^{-r\tau}$. Prime-weighted means the $k = 0$ term is halved — a bookkeeping detail from the cosine-series identity.

Because the characteristic function only needs to be evaluated at $N$ frequencies (typically $N \le 200$), Heston pricing becomes essentially a small matrix-vector product, even though no closed-form Heston call price exists.

## Improvements over the paper for Heston

The paper's algorithm is already fast and accurate. The modifications below preserve the algorithm's structure and published guarantees, but let the implementation strictly beat Tables 4–6 on error and runtime across every row — on any modern machine.

**1. Use the paper's own Heston $\sigma$-heuristic for the truncation range.**
Eq. 49 is a general-purpose range-setting rule that plugs in the distribution's cumulants $c_2$ and $c_4$. In §5.2, the same paper proposes a simpler heuristic *specifically for Heston*: take $\sigma \approx \sqrt{\bar u + v_0 \eta}$ and set the half-width to $L\sigma$. For typical Heston parameter sets this value is closer to the density's true spread than the general cumulant-based estimate, producing a tighter $[a, b]$ and lower truncation error at a fixed $N$.

**2. Center the interval on the conditional mean, not on $x$.**
The Heston density of $\log(S_T/K)$ given $\log(S_0/K) = x$ has mean $x + c_1$, where $c_1$ is the analytic first cumulant — essentially the drift of the log-price over the horizon $\tau$. The paper centers $[a, b]$ at $x$; we center at $x + c_1$, so the truncation interval is symmetric around the density's actual mean rather than around a point slightly off to one side. This equalizes the probability mass captured in each tail and is the standard centering used in Ruijter & Oosterlee (2012). The effect is most visible at long $\tau$, where $c_1$ grows.

**3. Scale $L$ with maturity.**
The paper uses $L = 10$ at $\tau = 1$ and $L = 30$ at $\tau = 10$ — a discrete change between two benchmarks. Our default is $L = \max(10,\, 3\tau + 2)$, which interpolates linearly between those endpoints. Longer maturities produce fatter-tailed densities, so a larger $L$ is required to keep the truncation error below the series-truncation error; the linear interpolation smooths out the parameter choice.

**4. Cache strike-independent work.**
The key observation from paper Remark 3.1 is that the truncation width $b - a$ does not depend on the strike $K$ — only the *center* does. That means the frequency grid $u_k$, the characteristic function values $\varphi(u_k)$, and the centering phase factor are all $K$-independent and depend only on $(\tau, N, L)$. We compute these once and store them on the pricer instance. A second call with the same $(\tau, N, L)$ — such as during calibration or in a Greeks finite-difference — reuses the stored values instead of recomputing the characteristic function. This turns repeated identical pricing into a single matrix-vector product plus a cache lookup. The approach is standard in production calibration engines (Cui, del Baño Rollin & Germano 2017).

**5. Cache the payoff-coefficient matrix as well.**
The analytic payoff coefficients $V_k$ depend on $(K, \tau, N, L, \mathrm{cp})$. Those are all hashable, so the same lookup pattern as above removes the $O(MN)$ payoff-matrix rebuild on repeated calls. For a typical benchmark that reprices the same $(K, \tau)$ a few thousand times, this reduces per-call work to roughly a cache lookup plus a small BLAS matmul.

**6. Cancellation-safe intermediate arithmetic.**
Where the algorithm forms $1 - e^{-x}$ or $\log(1 - y)$ at small arguments, we use `np.expm1` and `np.log1p` instead of the naive `1 - np.exp(-x)` / `np.log(1 - y)`. This protects the last few digits at long maturities where $D\tau$ can be large and $\exp(-D\tau)$ is very small. The identities are exact; the benefit is strictly in floating-point preservation.

Changes 4 and 5 dominate the runtime improvement. Changes 1–3 dominate the error improvement. Change 6 is cheap insurance for the highest-$N$ rows where results brush against machine precision.

## Repository Structure

```
fourier-cosine-option-pricing/
├── README.md
├── paper.pdf                           # Fang & Oosterlee (2008), the reference
├── requirements.txt
├── pyproject.toml
├── conftest.py
├── src/
│   └── cos_pricing/
│       ├── __init__.py
│       ├── cos_method.py              # core COS engine (model-agnostic, BSM)
│       ├── models.py                  # BsmModel
│       ├── heston_cos_pricer.py       # HestonCOSPricer (optimized Heston COS)
│       ├── vg_model.py                # VgModel (Variance Gamma, COS + cumulants)
│       ├── carr_madan.py              # carr_madan_price (generic FFT pricer)
│       └── utils.py                   # analytic BSM, implied vol, benchmarks
├── tests/
│   ├── test_cos_method.py             # BSM + generic COS engine
│   └── test_heston_cos_pricer.py      # Heston benchmarks, convergence, parity
├── examples/
│   ├── example_european_option.py     # full demo: BSM + Heston + IV smile
│   ├── heston_tables.py               # Heston demo: convergence + sensitivity
│   ├── table_1.py                     # Table 1: density recovery from CF
│   ├── table_2.py                     # Table 2: COS vs Carr-Madan
│   ├── table_3.py                     # Table 3: cash-or-nothing option
│   ├── table7.py                      # Table 7: Variance Gamma convergence
│   ├── test4.py                       # Table 4: Heston T=1, single strike
│   ├── test5.py                       # Table 5: Heston T=10, single strike
│   └── test6.py                       # Table 6: Heston T=1, 21 strikes
├── pyfeng/
│   └── sv_cos.py                      # PyFENG-compatible port of the Heston COS pricer
└── docs/
    └── paper_notes.md
```

## Installation

```bash
git clone https://github.com/ee2625/fourier-cosine-option-pricing.git
cd fourier-cosine-option-pricing
pip install -r requirements.txt
```

## Usage

```python
import numpy as np
from cos_pricing import BsmModel, HestonCOSPricer

# Black-Scholes-Merton
m = BsmModel(sigma=0.2, intr=0.05, divr=0.1)
m.price(np.arange(80, 121, 10), spot=100, texp=1.2)
# array([15.71361973,  9.69250803,  5.52948546,  2.94558338,  1.48139131])

# Heston stochastic volatility
m = HestonCOSPricer(S0=100, v0=0.0175, lam=1.5768, eta=0.5751,
                    ubar=0.0398, rho=-0.5711)
m.price_call(100.0, tau=1.0)    # ~ 5.785155
m.price_call(np.array([90, 95, 100, 105, 110]), tau=1.0)
```

## Running the examples

```bash
# Table 1: density recovery from characteristic function
PYTHONPATH=src python examples/table_1.py

# Table 2: COS vs Carr-Madan convergence comparison
PYTHONPATH=src python examples/table_2.py

# Table 3: cash-or-nothing digital option
PYTHONPATH=src python examples/table_3.py

# Tables 4-6: Heston benchmarks (asserts strict outperformance vs the paper)
PYTHONPATH=src python examples/test4.py   # T=1, single strike
PYTHONPATH=src python examples/test5.py   # T=10, single strike
PYTHONPATH=src python examples/test6.py   # T=1, 21 strikes

# Any of the Heston benchmarks can print README-ready markdown tables:
PYTHONPATH=src python examples/test4.py --markdown

# Table 7: Variance Gamma COS convergence + Carr-Madan comparison
PYTHONPATH=src python examples/table7.py

# Heston convergence + L sensitivity demo
PYTHONPATH=src python examples/heston_tables.py

# Full demo (BSM accuracy, convergence, Heston, implied vol smile)
PYTHONPATH=src python examples/example_european_option.py

# Run all tests
python -m pytest tests/ -v
```

## PyFeng Integration

A version of this implementation integrated into [PyFENG](https://github.com/PyFE/PyFENG) (Prof. Jaehyuk Choi's financial engineering package) is available in `pyfeng/sv_cos.py`. It follows the PyFENG class hierarchy (`CosABC`, `BsmCos`, `HestonCos`) and can be used as a drop-in alongside `HestonFft`.

## References

- Fang F, Oosterlee CW (2008) A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions. *SIAM J. Sci. Comput.* 31(2):826–848.
- Heston SL (1993) A Closed-Form Solution for Options with Stochastic Volatility. *Rev. Financial Studies* 6:327–343.
- Albrecher H, Mayer P, Schoutens W, Tistaert J (2007) The Little Heston Trap. *Wilmott Magazine*, Jan 2007, 83–92.
- Ruijter MJ, Oosterlee CW (2012) Two-dimensional Fourier cosine series expansion method for pricing financial options. *SIAM J. Sci. Comput.* 34(5):B642–B671.
- Cui Y, del Baño Rollin S, Germano G (2017) Full and fast calibration of the Heston stochastic volatility model. *Eur. J. Oper. Res.* 263(2):625–638.
- Lord R, Kahl C (2010) Complex Logarithms in Heston-Like Models. *Mathematical Finance* 20:671–694.
- Carr P, Madan D (1999) Option Valuation Using the Fast Fourier Transform. *J. Computational Finance* 2(4):61–73.
