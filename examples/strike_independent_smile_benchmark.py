"""
Benchmark strike-independent COS setup for volatility-smile pricing.

The point of the new setup API is not that vectorized COS was slow; it is
that the density-side grid and characteristic-function samples can be
computed once for a fixed model/expiry/range and then reused across many
strike vectors or calibration steps.

Run:
    PYTHONPATH=src python examples/strike_independent_smile_benchmark.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cos_pricing import CgmyModel, make_cos_smile_setup


def _mean_ms(fn, repeats):
    """Mean runtime in milliseconds."""
    fn()  # warm-up
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats * 1e3


def _benchmark_one(model, strikes, spot, texp, n_cos, repeats):
    fwd, df = model._fwd_df(spot, texp)
    cf = model.char_func(texp)
    trunc = model.trunc_range(texp)
    setup = make_cos_smile_setup(
        cf,
        texp,
        fwd,
        df,
        n_cos=n_cos,
        trunc_range=trunc,
    )

    def scalar_loop():
        return np.array([
            model.price(float(k), spot, texp, cp=1, n_cos=n_cos)
            for k in strikes
        ])

    def vector_price():
        return model.price(strikes, spot, texp, cp=1, n_cos=n_cos)

    def reusable_setup():
        return setup.price(strikes, cp=1)

    scalar_ms = _mean_ms(scalar_loop, repeats)
    vector_ms = _mean_ms(vector_price, repeats)
    setup_ms = _mean_ms(reusable_setup, repeats)

    ref = vector_price()
    setup_err = float(np.max(np.abs(reusable_setup() - ref)))
    scalar_err = float(np.max(np.abs(scalar_loop() - ref)))

    return {
        "n_strikes": len(strikes),
        "scalar_ms": scalar_ms,
        "vector_ms": vector_ms,
        "setup_ms": setup_ms,
        "scalar_speedup": scalar_ms / setup_ms,
        "vector_speedup": vector_ms / setup_ms,
        "setup_err": setup_err,
        "scalar_err": scalar_err,
    }


def main():
    model = CgmyModel(C=1.0, G=5.0, M=10.0, Y=1.5, intr=0.1, divr=0.0)
    spot = 100.0
    texp = 1.0
    n_cos = 512
    repeats = 300

    print("Strike-independent COS smile benchmark")
    print("Model: CGMY, C=1, G=5, M=10, Y=1.5, T=1, N=512")
    print(f"Repeats per row: {repeats}")
    print()
    print(
        f"{'strikes':>8}  {'scalar loop ms':>14}  {'vector ms':>10}  "
        f"{'setup ms':>10}  {'loop/setup':>11}  {'vector/setup':>12}  "
        f"{'max |setup-vector|':>20}"
    )
    print("-" * 100)

    for n_strikes in (5, 25, 101):
        strikes = np.linspace(60.0, 140.0, n_strikes)
        row = _benchmark_one(model, strikes, spot, texp, n_cos, repeats)
        print(
            f"{row['n_strikes']:8d}  "
            f"{row['scalar_ms']:14.4f}  "
            f"{row['vector_ms']:10.4f}  "
            f"{row['setup_ms']:10.4f}  "
            f"{row['scalar_speedup']:11.2f}x  "
            f"{row['vector_speedup']:12.2f}x  "
            f"{row['setup_err']:20.2e}"
        )

    print()
    print("Interpretation:")
    print("- Scalar loop rebuilds the COS density side once per strike.")
    print("- Vector price builds it once per call and is already efficient.")
    print("- Reusable setup builds it once total, then reuses it for the smile.")


if __name__ == "__main__":
    main()
