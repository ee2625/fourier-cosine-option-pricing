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

from .cos_method import cos_price
from .models import BsmModel, NormalCos
from .heston_cos_pricer import HestonCOSPricer
from .vg_model import VgModel
from .carr_madan import carr_madan_price
from .utils import bsm_price, bsm_impvol, convergence_table, benchmark_runtime
from .cgmy_model import CgmyModel

__all__ = [
    "cos_price",
    "BsmModel",
    "NormalCos",
    "HestonCOSPricer",
    "VgModel",
    "CgmyModel",
    "carr_madan_price",
    "bsm_price",
    "bsm_impvol",
    "convergence_table",
    "benchmark_runtime",
]
