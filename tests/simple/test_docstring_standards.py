"""Tests for docstring standards in the simple subpackage."""

import inspect

import pytest

import activations_plus.simple as simple
from activations_plus.simple import __all__


@pytest.mark.parametrize("func_name", __all__)
def test_docstring_standard(
    func_name,
):
    func = getattr(simple, func_name)
    docstring = inspect.getdoc(func)
    tags = [
        ".. math::",
        "Parameters\n",
        "Returns\n",
        "\n\nSource\n",
        ".. seealso::",
        "Example\n",
        ".. plot::",
    ]
    for tag in tags:
        assert tag in docstring, f"Docstring for {func.__name__} does not contain the required tag: {tag}"
        docstring = docstring.split(tag, 1)[-1]
