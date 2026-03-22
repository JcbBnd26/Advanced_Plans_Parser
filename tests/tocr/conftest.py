# tocr test configuration.
# Re-export root conftest helpers so ``from conftest import ...`` works.
import importlib.util as _ilu
from pathlib import Path as _Path

_root_conftest = _Path(__file__).resolve().parent.parent / "conftest.py"
_spec = _ilu.spec_from_file_location("_root_conftest", _root_conftest)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
for _name in dir(_mod):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_mod, _name)
del _name
