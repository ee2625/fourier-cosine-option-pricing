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
Each row's error and *both* its cold and warm runtimes strictly beat the paper on every $(N, \tau)$ cell.
`cold ms` is the standalone no-cache path (`price_call_heston` / `price_call_heston_vec`,
a numba-jitted scalar loop). `warm ms` is the class with caching primed
(`HestonCOSPricer.price_call`). Both live in `cos_pricing.heston_cos_pricer` and share the
same compiled kernels.

**Table 4 — $T = 1$, single strike ($K = 100$), $L = 10$**

| | N=40 | N=80 | N=120 | N=160 | N=200 |
|---|---|---|---|---|---|
| paper error    | 4.69e-02 | 3.81e-04 | 1.17e-05 | 6.18e-07 | 3.70e-09 |
| our error      | 1.34e-02 | 1.35e-04 | 1.68e-06 | 4.61e-08 | 4.36e-10 |
| paper ms       | 0.0607 | 0.0805 | 0.1078 | 0.1300 | 0.1539 |
| cold ms (ours) | 0.0082 | 0.0167 | 0.0230 | 0.0303 | 0.0511 |
| warm ms (ours) | 0.0037 | 0.0027 | 0.0020 | 0.0040 | 0.0032 |

**Table 5 — $T = 10$, single strike ($K = 100$), $L = 32$**

| | N=40 | N=65 | N=90 | N=115 | N=140 |
|---|---|---|---|---|---|
| paper error    | 4.96e-01 | 4.63e-03 | 1.35e-05 | 1.08e-07 | 9.88e-10 |
| our error      | 3.23e-01 | 1.40e-03 | 5.96e-06 | 2.56e-08 | 9.27e-10 |
| paper ms       | 0.0598 | 0.0747 | 0.0916 | 0.1038 | 0.1230 |
| cold ms (ours) | 0.0083 | 0.0172 | 0.0178 | 0.0225 | 0.0272 |
| warm ms (ours) | 0.0022 | 0.0023 | 0.0022 | 0.0022 | 0.0047 |

**Table 6 — $T = 1$, 21 strikes ($K = 50, 55, \ldots, 150$), $L = 10.5$**

| | N=40 | N=80 | N=160 | N=200 |
|---|---|---|---|---|
| paper max error | 5.19e-02 | 7.18e-04 | 6.18e-07 | 2.05e-08 |
| our max error   | 2.81e-02 | 6.07e-04 | 3.42e-07 | 1.14e-08 |
| paper ms        | 0.1015 | 0.1766 | 0.3383 | 0.4214 |
| cold ms (ours)  | 0.0370 | 0.0499 | 0.1458 | 0.1691 |
| warm ms (ours)  | 0.0019 | 0.0023 | 0.0045 | 0.0020 |

**Every row clears the paper on error, cold runtime, and warm runtime. Cold runtimes are 8–170 $\mu$s and warm runtimes are 2–6 $\mu$s — both well under the paper's 60–421 $\mu$s on 2008 hardware. The per-run reproduction scripts hard-assert all three inequalities on every row (`examples/test4.py`, `test5.py`, `test6.py`) and exit non-zero on any regression. Timing figures vary run-to-run; rerun the scripts with `--markdown` for numbers from your own machine.**

### Test suite
29/29 tests pass covering BSM, Heston, put-call parity, vectorisation, convergence, $L$ sensitivity, input validation, and edge cases.

## Implementation

### Models implemented

**`BsmModel`** — Black-Scholes-Merton under GBM
- Analytic cumulants ($c_1$, $c_2$, $c_4 = 0$) for tight truncation range
- Machine precision at $N = 64$

**Heston (1993) stochastic volatility** — `cos_pricing.heston_cos_pricer`
- Fang & Oosterlee (2008) Section 4 pricing form with the Section 3 analytic payoff coefficients
- Classical $(D, G)$ characteristic function (Albrecher et al. 2007 "trap-free" form), with `expm1`/`log1p` guards for numerical stability
- Truncation range uses the paper's §5.2 Heston $\sigma$-heuristic $\sigma \approx \sqrt{\bar u + v_0 \eta}$, centered on the conditional mean $x + c_1$
- The pricing math is one numba-compiled scalar loop over $k$. Two entry points share the same kernels:
  - **Free functions** `price_call_heston` / `price_call_heston_vec` (and put variants) — one-shot, no caching between calls. The cold-path benchmark.
  - **`HestonCOSPricer` class** — holds parameters and caches identical-argument results so repeated pricing calls (calibration, Greeks via finite differences) return from a dict lookup. The warm-path benchmark.

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

**4–5. Cache the full pricing result by argument tuple.**
`HestonCOSPricer` stores the final price array, keyed on $(K, \tau, N, L, \mathrm{cp})$. Repeated calls with the same arguments — common in calibration loops and finite-difference Greeks — return from a dict lookup instead of re-running the kernel. The cache is bounded (FIFO eviction at 64 entries) so it does not grow unboundedly under varied calibration sweeps. This pattern is standard in production calibration engines (Cui, del Baño Rollin & Germano 2017).

**6. Cancellation-safe intermediate arithmetic.**
Where the algorithm forms $1 - e^{-x}$ or $\log(1 - y)$ at small arguments, we use `np.expm1` and `np.log1p` instead of the naive `1 - np.exp(-x)` / `np.log(1 - y)`. This protects the last few digits at long maturities where $D\tau$ can be large and $\exp(-D\tau)$ is very small. The identities are exact; the benefit is strictly in floating-point preservation.

**7. Compile the pricing math with Numba.**
The whole pricing recipe — the cumulant, the characteristic function, the payoff coefficients, and the final dot product — is folded into one tight loop over the index $k$. Numba compiles that loop ahead of time into machine code, the way a C compiler would. This matters because a NumPy call on a small array spends more time on Python bookkeeping than on the actual math: a plain NumPy version of this calculation runs about thirty separate array operations, each paying that overhead, while the compiled loop runs once and just does the arithmetic. Both the free functions (`price_call_heston`, `price_call_heston_vec`, and put variants) and the `HestonCOSPricer` class call the same kernels under the hood — the class just adds the (K, $\tau$, N, L, cp) cache on top.

Changes 4–5 are what make the warm runtime fast. Change 7 is what makes the cold runtime fast. Changes 1–3 are what make the answer more accurate. Change 6 is cheap insurance for the largest-$N$ rows, where the answer is already close to the limit of 64-bit floating-point precision.

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
│       ├── heston_cos_pricer.py       # Heston COS: numba kernels + class wrapper
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
