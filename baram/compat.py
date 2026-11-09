"""Import-time compatibility shims, leaving original StrongSORT repository unchanged.

The upstream repository uses NumPy aliases and the removed scikit-learn linear
assignment import. We expose the old API before importing original modules.
"""
import collections
import collections.abc
import sys
import types
import numpy as np
from scipy.optimize import linear_sum_assignment


def enable():
    for name in ('Mapping','MutableMapping','Sequence','Iterable'):
        if not hasattr(collections,name):setattr(collections,name,getattr(collections.abc,name))
    for name,t in [('int',int),('float',float),('bool',bool)]:
        if not hasattr(np,name):setattr(np,name,t)
    module_name='sklearn.utils.linear_assignment_'
    if module_name not in sys.modules:
        # Protect against incompatible sklearn versions; scipy's Hungarian
        # algorithm uses identical cost minimization semantics.
        shim=types.ModuleType(module_name)
        def linear_assignment(cost):
            rows,cols=linear_sum_assignment(cost)
            return np.column_stack((rows,cols))
        shim.linear_assignment=linear_assignment
        sys.modules[module_name]=shim
