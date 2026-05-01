"""
Black-Scholes control variate helpers for COS pricing.

The correction has the form

    model_COS + (BS_exact - BS_COS),

where the Black-Scholes COS leg is evaluated on the same COS-style
truncation range as the target model whenever possible.
"""

import numpy as np

from .cos_method import cos_price
from .models import BsmModel
from .utils import bsm_price


def heston_average_variance_mean(v0, kappa, theta, texp):
    """
    Mean average variance under Heston over ``[0, T]``.

    E[Vbar_T] = theta + (v0 - theta) * (1 - exp(-kappa*T)) / (kappa*T).
    """
    T = float(texp)
    if T <= 0.0:
        raise ValueError(f"texp must be > 0, got {texp}")
    if kappa <= 0.0:
        raise ValueError(f"kappa must be > 0, got {kappa}")
    return float(theta + (v0 - theta) * (-np.expm1(-kappa * T)) / (kappa * T))


def heston_equivalent_bsm_vol(v0, kappa, theta, texp):
    """Equivalent BS volatility ``sqrt(E[Vbar_T])`` for Heston."""
    avg_var = heston_average_variance_mean(v0, kappa, theta, texp)
    if avg_var <= 0.0:
        raise ValueError(f"mean average variance must be > 0, got {avg_var}")
    return float(np.sqrt(avg_var))


def bsm_control_variate_adjustment(
    strike,
    spot,
    texp,
    sigma,
    intr=0.0,
    divr=0.0,
    cp=1,
    n_cos=128,
    trunc_range=None,
):
    """
    Return ``BS_exact - BS_COS`` for a Black-Scholes control variate.

    ``trunc_range`` is in the standard COS log-forward variable
    ``log(S_T/F)``.  Passing the target model's equivalent range is what
    makes the correction target the same truncation/series error.
    """
    bs = BsmModel(sigma=sigma, intr=intr, divr=divr)
    fwd, df = bs._fwd_df(spot, texp)
    bs_cos = cos_price(
        bs.char_func(texp),
        texp,
        strike,
        fwd,
        df,
        cp=cp,
        n_cos=n_cos,
        trunc_range=trunc_range if trunc_range is not None else bs.trunc_range(texp),
    )
    bs_exact = bsm_price(
        strike,
        spot,
        sigma,
        texp,
        intr=intr,
        divr=divr,
        cp=cp,
    )
    return bs_exact - bs_cos
