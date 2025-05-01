"""Tests to ensure we're not duplicating functionality already in PyTorch."""

import inspect

import pytest
import torch
import torch.nn.functional as functional

from activations_plus import simple
from activations_plus.simple import __all__ as simple_all


def get_torch_functions():
    """Get all activation functions in torch.nn.functional."""
    torch_funcs = []

    # Get all public functions from torch.nn.functional
    for name, obj in inspect.getmembers(functional):
        if not name.startswith("_") and callable(obj):
            torch_funcs.append(name.lower())

    return torch_funcs


def get_torch_modules():
    """Get all activation modules in torch.nn."""
    torch_modules = []

    # Get all public modules from torch.nn
    for name, obj in inspect.getmembers(torch.nn):
        if not name.startswith("_") and inspect.isclass(obj):
            torch_modules.append(name.lower())

    return torch_modules


# List of functions that are allowed to overlap with PyTorch
# These are reimplementations with potentially different behavior or enhanced functionality


@pytest.mark.parametrize("func_name", simple_all)
def test_no_duplicate_activation_functions(func_name):
    """Test that our activation functions don't overlap with PyTorch's built-in ones."""
    # Skip test for functions that are allowed to overlap

    torch_funcs = get_torch_functions()
    torch_modules = get_torch_modules()

    # Get the lowercase name without any suffix like "activation"
    base_name = func_name.lower().replace("_activation", "")

    # Check against both functional and module names
    assert base_name not in torch_funcs, f"{func_name} appears to duplicate a PyTorch functional API"
    assert base_name not in torch_modules, f"{func_name} appears to duplicate a PyTorch module"


@pytest.mark.parametrize("func_name", ["erf_act"])
def test_specific_activations_not_in_pytorch(func_name):
    """Specific test to ensure certain activations are not in PyTorch."""
    # Check functional API (lowercase)
    assert not hasattr(functional, func_name), f"{func_name} should not exist in PyTorch's functional API"

    # Check module API (capitalized)
    module_name = "".join(part.capitalize() for part in func_name.split("_"))
    assert not hasattr(torch.nn, module_name), f"{module_name} should not exist in PyTorch's module API"


@pytest.mark.parametrize("func_name", simple_all)
def test_all_activations_imported_correctly(func_name):
    """Test that all activation functions in __all__ are properly imported."""
    assert hasattr(simple, func_name), f"{func_name} listed in __all__ but not importable from simple"
