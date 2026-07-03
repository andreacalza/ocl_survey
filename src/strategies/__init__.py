#!/usr/bin/env python3
# Each strategy is imported defensively: some rely on older avalanche-lib
# APIs that may not exist on every installed version, and one incompatible
# strategy should not prevent the others from being used.
try:
    from .icarl import OnlineICaRL, OnlineICaRLLossPlugin
except ImportError as e:
    OnlineICaRL, OnlineICaRLLossPlugin = None, None
    _icarl_import_error = e

try:
    from .erace import ER_ACE
except ImportError as e:
    ER_ACE = None
    _erace_import_error = e

try:
    from .lwf import LwFPlugin
except ImportError as e:
    LwFPlugin = None
    _lwf_import_error = e

try:
    from .agem import AGEMPlugin
except ImportError as e:
    AGEMPlugin = None
    _agem_import_error = e
