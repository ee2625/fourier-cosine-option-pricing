"""
Pytest configuration shared across the test suite.

The local ``pyfeng/`` folder in this repo has no ``__init__.py``, so
Python treats it as a namespace package. When the repo root is on
``sys.path`` (e.g. when pytest is invoked from the repo root), ``import
pyfeng`` resolves to this local folder -- which lacks ``opt_abc.py``,
``sv_abc.py``, ``heston.py``, etc. -- and the relative imports inside
``sv_cos.py`` / ``sv_heston_cos.py`` fail with ``ModuleNotFoundError``.

This conftest fixes the resolution by:

1. Stripping the repo root from ``sys.path`` so the local ``pyfeng/``
   namespace is no longer discoverable as a package.
2. Importing the installed ``pyfeng`` (if present), then appending the
   repo's ``pyfeng/`` directory to its ``__path__`` so our new modules
   (``sv_cos.py``, ``sv_heston_cos.py``) resolve as
   ``pyfeng.sv_cos`` / ``pyfeng.sv_heston_cos`` while ``pyfeng.heston``,
   ``pyfeng.opt_abc``, etc. continue to resolve to the install.

Cross-check tests still need ``src/`` on ``sys.path`` to import
``cos_pricing``; that's added below as well.
"""

import importlib
import os
import sys


_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYFENG_LOCAL = os.path.join(_REPO, "pyfeng")
_SRC = os.path.join(_REPO, "src")


def _strip_repo_from_sys_path():
    """Remove the repo root so the local ``pyfeng/`` namespace doesn't shadow."""
    sys.path[:] = [
        p for p in sys.path
        if os.path.abspath(p) not in (_REPO, "")
    ]


def _overlay_local_pyfeng():
    """Append local pyfeng/ to the installed package's __path__, if installed."""
    try:
        import pyfeng
    except ImportError:
        return
    # Only overlay if the install is real (has expected submodules).
    install_paths = [p for p in pyfeng.__path__ if os.path.abspath(p) != _PYFENG_LOCAL]
    if not install_paths:
        return
    if _PYFENG_LOCAL not in pyfeng.__path__:
        pyfeng.__path__.append(_PYFENG_LOCAL)
    importlib.invalidate_caches()


def _add_src_to_sys_path():
    """Make ``cos_pricing`` (the source pricer) importable for cross-checks."""
    if os.path.isdir(_SRC) and _SRC not in sys.path:
        sys.path.insert(0, _SRC)


_strip_repo_from_sys_path()
_overlay_local_pyfeng()
_add_src_to_sys_path()
