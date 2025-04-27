import pathlib


def test_examples_match_functions():
    from activations_plus.simple import __all__ as simple_all

    examples_dir = pathlib.Path(__file__).parent.parent.parent / "examples"

    # Only test actual example scripts, not __init__.py files
    example_files = [f for f in examples_dir.rglob("*.py") if f.name != "__init__.py"]
    assert len(example_files) == len(simple_all), "Mismatch between example files and functions in simple module."
