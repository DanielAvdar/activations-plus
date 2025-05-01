import importlib
import pathlib
import sys
from unittest import mock

import pytest

EXAMPLES_DIR = pathlib.Path(__file__).parent.parent.parent / "examples"

# Only test actual example scripts, not __init__.py files
example_files = [f for f in EXAMPLES_DIR.rglob("*.py") if f.name != "__init__.py"]


@pytest.mark.parametrize("example_path", example_files)
def test_example_runs_without_show(example_path):
    """Test that each example script runs without error, mocking fig.show()."""
    rel_path = example_path.relative_to(EXAMPLES_DIR.parent)
    module_name = str(rel_path.with_suffix("")).replace("/", ".").replace("\\", ".")
    with mock.patch("matplotlib.figure.Figure.show", autospec=True):
        importlib.invalidate_caches()
        if module_name in sys.modules:
            del sys.modules[module_name]
        importlib.import_module(module_name)
