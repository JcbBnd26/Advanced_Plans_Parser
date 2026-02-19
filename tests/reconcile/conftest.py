"""Re-export shared fixtures from the root conftest."""

import importlib.util
import pathlib
import sys

_root_conftest_path = str(
    pathlib.Path(__file__).resolve().parent.parent / "conftest.py"
)
_spec = importlib.util.spec_from_file_location("_root_conftest", _root_conftest_path)
_rc = importlib.util.module_from_spec(_spec)
sys.modules["_root_conftest"] = _rc
_spec.loader.exec_module(_rc)
make_box = _rc.make_box  # noqa: F401
