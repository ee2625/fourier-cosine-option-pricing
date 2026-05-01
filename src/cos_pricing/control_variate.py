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


def variance_equivalent_bsm_vol(log_return_variance, texp):
    """Equivalent BS volatility from a log-return variance cumulant."""
    T = float(texp)
    if T <= 0.0:
        raise ValueError(f"texp must be > 0, got {texp}")
    variance = float(log_return_variance)
    if variance <= 0.0:
        raise ValueError(f"log_return_variance must be > 0, got {variance}")
    return float(np.sqrt(variance / T))


def _vol_from_variance_rate(variance_rate):
    variance_rate = float(np.real(variance_rate))
    if not np.isfinite(variance_rate) or variance_rate <= 0.0:
        raise ValueError(f"equivalent BS variance rate must be > 0, got {variance_rate}")
    return float(np.sqrt(variance_rate))


def joshi_yang_real_axis_bsm_vol(mgf, texp, eps=1e-5):
    """
    Joshi-Yang real-axis BS volatility selector, Eq. (3.5).

    The paper matches first derivatives of the target and BS characteristic
    functions at ``-i``.  In MGF notation for ``X = log(S_T/F)``, this is

        sigma^2 = 2 * K'(1) / T,

    where ``K(u) = log(E[exp(u X)])``.  This corresponds to matching
    ``E[(S_T/F) log(S_T/F)]``.
    """
    T = float(texp)
    if T <= 0.0:
        raise ValueError(f"texp must be > 0, got {texp}")
    if eps <= 0.0:
        raise ValueError(f"eps must be > 0, got {eps}")

    kp = (
        np.log(complex(mgf(1.0 + eps)))
        - np.log(complex(mgf(1.0 - eps)))
    ) / (2.0 * eps)
    return _vol_from_variance_rate(2.0 * kp.real / T)


def joshi_yang_contour_bsm_vol(mgf, texp, eta=0.5):
    """
    Joshi-Yang contour BS volatility selector, Eq. (3.4).

    For a contour with imaginary part ``eta`` not equal to 0 or 1, the BS
    and target characteristic functions are matched at ``i(eta - 1)``.
    In MGF notation this is

        sigma^2 = 2 * log(M(1 - eta)) / (eta * (eta - 1) * T).

    The paper's common Hermitian contour is ``eta = 0.5``.
    """
    T = float(texp)
    eta = float(eta)
    if T <= 0.0:
        raise ValueError(f"texp must be > 0, got {texp}")
    if np.isclose(eta, 0.0) or np.isclose(eta, 1.0):
        raise ValueError("eta must not be 0 or 1 for Eq. (3.4)")

    log_m = np.log(complex(mgf(1.0 - eta))).real
    return _vol_from_variance_rate(2.0 * log_m / (eta * (eta - 1.0) * T))


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
