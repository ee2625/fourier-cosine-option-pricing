"""
cos_pricing
===========
Fourier-Cosine (COS) method for European option pricing.

Reference:
    Fang F, Oosterlee CW (2008) A Novel Pricing Method for European Options
    Based on Fourier-Cosine Series Expansions.
    SIAM J. Sci. Comput. 31(2):826-848.
    https://doi.org/10.1137/080718061
"""

from .cos_method import CosSmileSetup, cos_price, cos_price_smile, make_cos_smile_setup
from .models import BsmModel
from .heston_cos_pricer import (
    HestonCOSPricer,
    price_call_heston,
    price_put_heston,
    price_call_heston_vec,
    price_put_heston_vec,
)
from .vg_model import VgModel
from .carr_madan import carr_madan_price
from .lewis import lewis_price
from .frft import frft_price
from .bermudan import BermudanCosBSM
from .utils import bsm_price, bsm_impvol, convergence_table, benchmark_runtime
from .cgmy_model import CgmyModel
from .cos_range import (
    central_moment_from_cumulants,
    jp_markov_half_width,
    jp_markov_range,
)
from .control_variate import (
    bsm_control_variate_adjustment,
    heston_average_variance_mean,
    heston_equivalent_bsm_vol,
)

__all__ = [
    "cos_price",
    "cos_price_smile",
    "make_cos_smile_setup",
    "CosSmileSetup",
    "BsmModel",
    "HestonCOSPricer",
    "price_call_heston",
    "price_put_heston",
    "price_call_heston_vec",
    "price_put_heston_vec",
    "VgModel",
    "CgmyModel",
    "carr_madan_price",
    "lewis_price",
    "frft_price",
    "BermudanCosBSM",
    "bsm_price",
    "bsm_impvol",
    "convergence_table",
    "benchmark_runtime",
    "central_moment_from_cumulants",
    "jp_markov_half_width",
    "jp_markov_range",
    "bsm_control_variate_adjustment",
    "heston_average_variance_mean",
    "heston_equivalent_bsm_vol",
]
