"""Compatibility alias for :mod:`melody_features.idyom.interface`."""

import sys

from .idyom import interface as _interface

sys.modules[__name__] = _interface
