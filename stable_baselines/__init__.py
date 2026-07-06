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
    _candidate = os.path.join(_root, "stable_baselines")
    if os.path.isdir(_candidate) and _candidate not in __path__:
        __path__.append(_candidate)

from . import logger  # noqa: E402,F401
