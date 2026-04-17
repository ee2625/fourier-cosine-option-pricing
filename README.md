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
- Computational efficiency (accuracy vs. N, runtime scaling)

## Key Results

### Density recovery from characteristic function (Table 1 reproduction)
f(x) = N(0,1), recovered from its CF via cosine expansion on [-10, 10]

| | N=4 | N=8 | N=16 | N=32 | N=64 |
|---|---|---|---|---|---|
| max error    | 2.54e-01 | 1.08e-01 | 7.18e-03 | 4.04e-07 | 3.89e-16 |
| cpu time (sec) | ~0.0000 | ~0.0000 | ~0.0000 | ~0.0000 | ~0.0000 |

![Table 1](examples/table_1.png)

**Exponential convergence: errors decrease ~10x per doubling of N, reaching machine precision at N=64. This demonstrates the mathematical foundation of the entire COS method.**

### BSM model — COS vs Carr-Madan (Table 2 reproduction)
σ=0.25, r=0.1, q=0, T=0.1, S=100, K=80/100/120

| | | N=32 | N=64 | N=128 | N=256 | N=512 |
|---|---|---|---|---|---|---|
| COS | msec | 0.1066 | 0.1073 | 0.1199 | 0.1508 | 0.2039 |
| | max error | 2.43e-07 | 1.81e-14 | 1.81e-14 | 1.81e-14 | 1.81e-14 |
| Carr-Madan | msec | 0.0428 | 0.0429 | 0.0466 | 0.0536 | 0.0688 |
| | max error | 1.29e+02 | 2.57e+02 | 4.80e+01 | 1.29e+00 | 1.29e+00 |

![Table 2](examples/table_2.png)

**COS reaches machine precision at N=64. Carr-Madan requires N>512 for comparable accuracy.**

### Cash-or-nothing digital option — COS (Table 3 reproduction)
σ=0.2, r=0.05, q=0, T=0.1, S=100, K=120

| | N=40 | N=60 | N=80 | N=100 | N=120 | N=140 |
|---|---|---|---|---|---|---|
| error          | 3.67e-11 | 2.87e-16 | 2.87e-16 | 2.87e-16 | 2.87e-16 | 2.87e-16 |
| cpu time (msec) | 0.0283 | 0.0297 | 0.0317 | 0.0326 | 0.0340 | 0.0348 |

![Table 3](examples/table_3.png)

**Exponential convergence holds for discontinuous payoffs when analytic coefficients are used — confirming Theorem 3.1 of the paper.**

### Table 4 reproduction
| | N=40 | N=80 | N=120 | N=160 | N=200 |
|---|---|---|---|---|---|
| paper error    | 4.69e-02 | 3.81e-04 | 1.17e-05 | 6.18e-07 | 3.70e-09 |
| our error      | 1.34e-02 | 1.35e-04 | 1.68e-06 | 4.61e-08 | 4.36e-10 |
| paper ms       | 0.0607 | 0.0805 | 0.1078 | 0.1300 | 0.1539 |
| cold ms (ours) | 0.1114 | 0.1293 | 0.1440 | 0.1622 | 0.1773 |
| warm ms (ours) | 0.0069 | 0.0069 | 0.0073 | 0.0070 | 0.0071 |

### Table 5 reproduction
| | N=40 | N=65 | N=90 | N=115 | N=140 |
|---|---|---|---|---|---|
| paper error    | 4.96e-01 | 4.63e-03 | 1.35e-05 | 1.08e-07 | 9.88e-10 |
| our error      | 3.23e-01 | 1.40e-03 | 5.96e-06 | 2.56e-08 | 9.42e-10 |
| paper ms       | 0.0598 | 0.0747 | 0.0916 | 0.1038 | 0.1230 |
| cold ms (ours) | 0.1195 | 0.1363 | 0.1398 | 0.1598 | 0.1620 |
| warm ms (ours) | 0.0077 | 0.0077 | 0.0075 | 0.0078 | 0.0076 |

### Table 6 reproduction
| | N=40 | N=80 | N=160 | N=200 |
|---|---|---|---|---|
| paper max error | 5.19e-02 | 7.18e-04 | 6.18e-07 | 2.05e-08 |
| our max error   | 2.81e-02 | 6.07e-04 | 3.42e-07 | 1.14e-08 |
| paper ms        | 0.1015 | 0.1766 | 0.3383 | 0.4214 |
| warm ms (ours)  | 0.0078 | 0.0079 | 0.0083 | 0.0085 |


### Test suite
29/29 tests pass covering BSM, Heston, put-call parity, vectorisation, convergence, L sensitivity, input validation, and edge cases.

## Implementation

### Models implemented

**`BsmModel`** — Black-Scholes-Merton under GBM
- Analytic cumulants (c1, c2, c4=0) for tight truncation range
- Machine precision at N=64

**`HestonCOSPricer`** — Heston (1993) stochastic volatility, paper-faithful COS
- Fang & Oosterlee (2008) Section 4 form: `x = log(S0/K)` centering, classical (D, G) CF, Section 3 analytic payoff coefficients
- Analytic cumulants from F&O Appendix A set the truncation width via Eq. 49
- Common-subexpression-optimized characteristic function — one `sqrt`, one `log`, one `exp(-Dτ)` per call
- Default `L=12` reaches paper-grade accuracy at both T=1 and T=10 with N=160

### Core formula

The COS price of a European option is (Eq. 21 of the paper):

```
V(x, t) = df * F * sum'_{k=0}^{N-1} Re[phi(k*pi/(b-a)) * exp(-i*k*pi*a/(b-a))] * V_k
```

where:
- `phi(u)` is the characteristic function of log(S_T/F)
- `V_k` are analytic payoff coefficients (chi and psi integrals, Eqs. 22-23)
- `[a, b]` is the truncation range set from cumulants (Eq. 5.2)
- `sum'` denotes the prime sum (k=0 term gets weight 1/2)

The dominant cost is one (M × N) matrix-vector product for M strikes simultaneously.

## Repository Structure

```
fourier-cosine-option-pricing/
├── README.md
├── requirements.txt
├── pyproject.toml
├── conftest.py
├── src/
│   └── cos_pricing/
│       ├── __init__.py
│       ├── cos_method.py              # core COS engine (model-agnostic, BSM)
│       ├── models.py                  # BsmModel
│       ├── heston_cos_pricer.py       # HestonCOSPricer (paper-faithful Section 4 form)
│       └── utils.py                   # analytic BSM, implied vol, benchmarks
├── tests/
│   ├── test_cos_method.py             # BSM + generic COS engine
│   └── test_heston_cos_pricer.py      # Heston benchmarks, convergence, parity
├── examples/
│   ├── example_european_option.py     # full demo: BSM + Heston + IV smile
│   ├── heston_cos_pricer.py           # Heston paper benchmark + convergence tables
│   ├── table_1.py                     # Table 1: density recovery from CF
│   ├── GBM_cos_vs_carr_madan.py       # Table 2: COS vs Carr-Madan
│   └── table_3.py                     # Table 3: cash-or-nothing option
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

# Heston stochastic volatility (paper-faithful COS, Fang & Oosterlee 2008 §4)
m = HestonCOSPricer(S0=100, v0=0.0175, lam=1.5768, eta=0.5751,
                    ubar=0.0398, rho=-0.5711)
m.price_call(100.0, tau=1.0)    # ≈ 5.785155
m.price_call(np.array([90, 95, 100, 105, 110]), tau=1.0)
```

## Running the examples

```bash
# Table 1: density recovery from characteristic function
PYTHONPATH=src python examples/table_1.py

# Table 2: COS vs Carr-Madan convergence comparison
PYTHONPATH=src python examples/GBM_cos_vs_carr_madan.py

# Table 3: cash-or-nothing digital option
PYTHONPATH=src python examples/table_3.py

# Heston COS paper benchmark (T=1 and T=10), convergence and L sensitivity
PYTHONPATH=src python examples/heston_cos_pricer.py

# Full demo (BSM accuracy, convergence, Heston, implied vol smile)
PYTHONPATH=src python examples/example_european_option.py

# Run all tests
python -m pytest tests/ -v
```

## PyFeng Integration

A version of this implementation integrated into [PyFENG](https://github.com/PyFE/PyFENG) (Prof. Jaehyuk Choi's financial engineering package) is available in `sv_cos.py`. It follows the PyFENG class hierarchy (`CosABC`, `BsmCos`, `HestonCos`) and can be used as a drop-in alongside `HestonFft`.

## References

- Fang F, Oosterlee CW (2008) A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions. *SIAM J. Sci. Comput.* 31(2):826–848.
- Heston SL (1993) A Closed-Form Solution for Options with Stochastic Volatility. *Rev. Financial Studies* 6:327–343.
- Lord R, Kahl C (2010) Complex Logarithms in Heston-Like Models. *Mathematical Finance* 20:671–694.
- Carr P, Madan D (1999) Option Valuation Using the Fast Fourier Transform. *J. Computational Finance* 2(4):61–73.
