"""
Lévy-process European option pricing via the Fourier-Cosine (COS) method.

Two pricers, each a sibling of an existing PyFENG FFT pricer:

* ``VarGammaCos`` -- Variance Gamma.  Inherits the MGF, parameter
  signature, validation, and the omega-correction precomputation from
  ``VarGammaFft``.  Adds analytic cumulants from F&O (2008) Table 11
  so the COS truncation range from ``CosABC`` is exact rather than
  finite-difference.
* ``CgmyCos`` -- CGMY infinite-activity Lévy.  Inherits the MGF,
  parameter signature, validation, and precomputed
  ``C*Gamma(-Y), M^Y, G^Y, kappa(1)`` constants from ``CgmyFft``.
  Overrides the truncation range with the Y-dependent heuristic from
  F&O (2008) Section 5.4 because CGMY moments diverge as ``Y -> 2``.

Both classes use multiple inheritance ``(<Fft sibling>, CosABC)``.  MRO
would otherwise route ``price()`` through ``FftABC``, so each class
explicitly binds ``price = CosABC.price`` to use the COS dot product
instead of the FFT integration.  All other COS hooks
(``make_smile_setup``, ``price_smile``, ``_truncation_range``,
``_cos_grid``, ``_density_coefficients``,
``_vanilla_payoff_coefficients``) come from ``CosABC`` via the MRO.

References:
    - Madan DB, Carr PP, Chang EC (1998) European Finance Review 2:79-105.
      https://doi.org/10.1023/A:1009703431535
    - Carr P, Geman H, Madan DB, Yor M (2002) Journal of Business 75:305-332.
      https://doi.org/10.1086/338705
    - Fang F, Oosterlee CW (2008) SIAM J. Sci. Comput. 31:826-848.
      https://doi.org/10.1137/080718061
"""

import numpy as np

from .cos_range import cgmy_cumulants, vg_cumulants
from .sv_cos import CosABC
from .sv_fft import VarGammaFft, CgmyFft


__all__ = ["VarGammaCos", "CgmyCos"]


class VarGammaCos(VarGammaFft, CosABC):
    """
    Variance Gamma (VG) European option pricing via the COS method.

    The log price under the VG model is a drifted Brownian motion
    time-changed by a Gamma subordinator (Madan, Carr, Chang 1998):
    ``log(S_T/F) = theta*G_T + sigma*W_{G_T} + omega*T``, with
    ``G_T ~ Gamma(T/vov, vov)`` and the martingale correction
    ``omega = log(1 - theta*vov - 0.5*sigma^2*vov) / vov``.

    Inherits the MGF body and the parameter signature from ``VarGammaFft``
    -- single CF source of truth across the COS and FFT pricers.  Adds
    analytic cumulants from F&O 2008 Table 11 so ``CosABC._truncation_range``
    uses exact c2, c4 instead of the default finite-difference fallback.

    Parameters (inherited from ``VarGammaFft``; ``vov`` plays the role of
    nu, the Gamma variance rate):

        sigma   Brownian volatility, must be > 0.
        vov     Gamma variance rate (= nu), must be > 0.
        theta   Brownian drift; controls skewness.  Default 0.0.
        rho, mr Not used; inherited fillers from SvABC.
        intr, divr, is_fwd  Standard PyFENG.

    The constraint ``1 - theta*vov - 0.5*sigma^2*vov > 0`` must hold for
    omega to be real-valued; otherwise ``VarGammaFft.__init__`` raises
    ``ValueError``.

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.VarGammaCos(sigma=0.12, theta=-0.14, vov=0.2,
        ...                    intr=0.1, divr=0.0)
        >>> m.price(np.array([90.0, 100.0, 110.0]), 100.0, 1.0)
    """

    # MRO routes price() through FftABC by default; rebind to the COS path.
    price = CosABC.price
    make_smile_setup = CosABC.make_smile_setup
    price_smile = CosABC.price_smile

    def __init__(self, sigma, vov, theta=0.0, **kwargs):
        """Pre-validate sigma > 0, vov > 0 before delegating to VarGammaFft.

        VarGammaFft only enforces the omega constraint
        ``1 - theta*vov - 0.5*sigma^2*vov > 0``; it accepts sigma <= 0 and
        vov <= 0 silently, which produces NaNs downstream.  We surface those
        as ValueError to match the rest of the COS-family input validation.
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if vov <= 0:
            raise ValueError(f"vov (Gamma variance rate) must be > 0, got {vov}")
        super().__init__(sigma=sigma, vov=vov, theta=theta, **kwargs)

    def mgf_logprice(self, uu, texp):
        """Wrap VarGammaFft.mgf_logprice to handle scalar inputs.

        Upstream VarGammaFft.mgf_logprice uses np.exp(out=rv); on numpy 2.x
        rv must be a writable array, not a 0-d scalar.  We promote scalar
        inputs to length-1 arrays and unwrap the result to preserve the
        scalar-in-scalar-out contract callers expect.
        """
        uu_arr = np.atleast_1d(np.asarray(uu) + 0j)
        out = super().mgf_logprice(uu_arr, texp)
        if np.ndim(uu) == 0:
            return complex(np.asarray(out).flat[0])
        return out

    def _cumulants(self, texp):
        """Analytic VG cumulants of log(S_T/F) -- F&O 2008 Table 11.

        ``c1 = T*(omega + theta)``,
        ``c2 = (sigma^2 + vov*theta^2) * T``,
        ``c4 = 3*(sigma^4*vov + 2*theta^4*vov^3 + 4*sigma^2*theta^2*vov^2) * T``,
        with ``omega = log(1 - theta*vov - 0.5*sigma^2*vov) / vov``.

        Reuses the inherited ``_mgf1_correction = -log(arg)`` so
        ``omega = -_mgf1_correction / vov`` -- single source of truth at
        the Python level.

        Returns:
            ``(c1, c2, 0.0, c4)``: 4-tuple matching ``CosABC._cumulants``.
        """
        T = float(texp)
        sig2 = self.sigma**2
        omega = -self._mgf1_correction / self.vov
        c1 = T * (omega + self.theta)
        c2 = (sig2 + self.vov * self.theta**2) * T
        c4 = 3.0 * (sig2**2 * self.vov
                    + 2.0 * self.theta**4 * self.vov**3
                    + 4.0 * sig2 * self.theta**2 * self.vov**2) * T
        return float(c1), float(c2), 0.0, float(c4)

    def _jp_cumulants(self, texp, order):
        """Analytic VG cumulants through ``order`` for the JP range."""
        return vg_cumulants(
            sigma=self.sigma,
            theta=self.theta,
            nu=self.vov,
            texp=texp,
            order=order,
        )


class CgmyCos(CgmyFft, CosABC):
    """
    CGMY infinite-activity Lévy European option pricing via the COS method.

    The CGMY process (Carr, Geman, Madan, Yor 2002) is a four-parameter
    pure-jump Lévy process with Lévy measure
    ``nu(dx) = C * [exp(-M*x)/x^(1+Y) * 1{x>0} + exp(G*x)/|x|^(1+Y) * 1{x<0}] dx``
    and per-unit-time CGF
    ``kappa(u) = C*Gamma(-Y) * [(M-u)^Y - M^Y + (G+u)^Y - G^Y]``.

    Inherits the MGF body and the parameter signature from ``CgmyFft`` --
    single CF source of truth across the COS and FFT pricers.  Overrides
    ``_truncation_range`` with the Y-dependent heuristic from F&O 2008
    Section 5.4 because CGMY moments diverge as ``Y -> 2``, making the
    cumulant-based default of ``CosABC`` unreliable for high Y.

    Parameters (inherited from ``CgmyFft``):

        C   Overall jump activity / scale, must be > 0.
        G   Left-tail decay rate, must be > 0.  Larger G => thinner left tail.
        M   Right-tail decay rate, must be > 1.  M > 1 is required for the
            martingale correction kappa(1) to be finite.
        Y   Fine-structure index, must be < 2.  Y < 0 is finite activity
            (compound Poisson); Y in [0, 1) is infinite activity, finite
            variation; Y in [1, 2) is infinite variation.  Y in {0, 1, 2}
            are singular for Gamma(-Y); avoid those exact values.
        intr, divr, is_fwd  Standard PyFENG.

    Note:
        The source ``cos_pricing.CgmyModel`` defaults to L = 10, whereas
        ``CosABC`` defaults to L = 12.  Set ``m.L = 10`` to reproduce
        source prices exactly.

    Examples:
        >>> import numpy as np
        >>> import pyfeng as pf
        >>> m = pf.CgmyCos(C=1.0, G=5.0, M=10.0, Y=0.5)
        >>> m.price(np.arange(80, 121, 10), 100.0, 1.0)
    """

    # MRO routes price() through FftABC by default; rebind to the COS path.
    price = CosABC.price
    make_smile_setup = CosABC.make_smile_setup
    price_smile = CosABC.price_smile

    def mgf_logprice(self, uu, texp):
        """Wrap CgmyFft.mgf_logprice to handle scalar inputs.

        Same numpy-2.x scalar-vs-array workaround as VarGammaCos.mgf_logprice.
        """
        uu_arr = np.atleast_1d(np.asarray(uu) + 0j)
        out = super().mgf_logprice(uu_arr, texp)
        if np.ndim(uu) == 0:
            return complex(np.asarray(out).flat[0])
        return out

    def _truncation_range(self, texp):
        """COS truncation range -- F&O 2008 Section 5.4 Y-dependent heuristic.

        Returns ``[-100, 20]`` for Y close to 1.98 (F&O's special case),
        otherwise ``[-L*Y, L*Y]`` using ``self.L``.
        """
        if np.isclose(self.Y, 1.98):
            return -100.0, 20.0
        return -self.L * self.Y, self.L * self.Y

    def _jp_cumulants(self, texp, order):
        """Analytic CGMY cumulants through ``order`` for the JP range."""
        return cgmy_cumulants(
            C=self.C,
            G=self.G,
            M=self.M,
            Y=self.Y,
            texp=texp,
            order=order,
        )
