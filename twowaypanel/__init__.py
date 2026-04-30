# -*- coding: utf-8 -*-
"""
twowaypanel
===========

twowaypanel is a Python package for bias-corrected estimation and inference
in nonlinear panel data models with two-way (additive individual and time) fixed effects.

Quick links
-----------
- Documentation: https://twowaypanel.readthedocs.io/
- Source code & replication materials (GitHub): https://github.com/zizhongyan/twowaypanel

Main entry point
----------------
Most users will start with :func:`twowaypanel.fit`, which fits two-way fixed-effects
nonlinear panel models (binary logit/probit, multinomial logit, ordered logit),
optionally applying likelihood-based and/or analytical bias correction, and can also
compute average partial effects (APEs).

Notes
-----
- For full replication workflow of Yan et al. (2026), see the replication notebooks in the GitHub repository.
- For usage examples and model-by-model demonstrations, see the documentation tutorials and examples gallery.


Version
-------
{version}
"""

__version__ = "0.9.4"

# Inject version into the module docstring shown by help(twowaypanel)
__doc__ = (__doc__ or "").format(version=__version__)

from .api.panelModels import fit
from . import database
from . import demo
from .lib.torchmin import minimize as minimize2

