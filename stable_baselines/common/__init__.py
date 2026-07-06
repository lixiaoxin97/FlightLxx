import os
import site
import sysconfig

from rl.tf_compat import enable_tf1_compat


enable_tf1_compat()

_search_roots = []
_search_roots.extend(site.getsitepackages())
_search_roots.append(site.getusersitepackages())
_search_roots.append(sysconfig.get_paths().get("purelib", ""))

for _root in _search_roots:
    _candidate = os.path.join(_root, "stable_baselines", "common")
    if os.path.isdir(_candidate) and _candidate not in __path__:
        __path__.append(_candidate)

from stable_baselines.common.base_class import (  # noqa: E402,F401
    ActorCriticRLModel,
    BaseRLModel,
    OffPolicyRLModel,
    SetVerbosity,
    TensorboardWriter,
)
from stable_baselines.common.math_util import explained_variance  # noqa: E402,F401
