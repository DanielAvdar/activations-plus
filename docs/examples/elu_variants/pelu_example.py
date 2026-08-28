"""Example demonstrating the PELU activation function."""

import matplotlib.pyplot as plt
import torch

from activations_plus.simple import pelu


def main() -> None:
    """Plot the PELU activation function with different parameter values."""
    x = torch.linspace(-5, 5, 1000)

    # Different parameter combinations
    params = [
        {"alpha": 1.0, "beta": 1.0, "label": "α=1.0, β=1.0 (default)"},
        {"alpha": 1.5, "beta": 1.0, "label": "α=1.5, β=1.0"},
        {"alpha": 1.0, "beta": 1.5, "label": "α=1.0, β=1.5"},
        {"alpha": 1.5, "beta": 1.5, "label": "α=1.5, β=1.5"},
    ]

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot PELU with different parameter combinations
    for param in params:
        y_pelu = pelu(x, alpha=param["alpha"], beta=param["beta"])
        plt.plot(x.numpy(), y_pelu.numpy(), label=param["label"], linewidth=2)

    # Add vertical and horizontal lines at origin
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.axvline(x=0, color="k", linestyle="-", alpha=0.3)

    # Configure the plot
    plt.grid(True, alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("PELU Activation Function with Different Parameters")
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
