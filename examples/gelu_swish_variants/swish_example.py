"""Example demonstrating the Swish activation function."""

import matplotlib.pyplot as plt
import torch

from activations_plus.simple import swish


def main() -> None:
    """Plot the Swish activation function with different beta values and SiLU."""
    x = torch.linspace(-5, 5, 1000)

    # Compute Swish with different beta values
    y_beta_0_5 = swish(x, beta=0.5)
    y_beta_1_0 = swish(x, beta=1.0)  # This is equivalent to SiLU
    y_beta_2_0 = swish(x, beta=2.0)

    # Convert to numpy for plotting
    x_np = x.numpy()
    y_beta_0_5_np = y_beta_0_5.numpy()
    y_beta_1_0_np = y_beta_1_0.numpy()
    y_beta_2_0_np = y_beta_2_0.numpy()

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_np, y_beta_0_5_np, label="Swish (β=0.5)", linewidth=2)
    plt.plot(x_np, y_beta_1_0_np, label="Swish (β=1.0)", linewidth=2)
    plt.plot(x_np, y_beta_2_0_np, label="Swish (β=2.0)", linewidth=2)

    # Add reference line y=x
    plt.plot(x_np, x_np, label="y=x", linestyle=":", color="gray", alpha=0.7)

    # Add vertical and horizontal lines at origin
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.axvline(x=0, color="k", linestyle="-", alpha=0.3)

    # Configure the plot
    plt.grid(True, alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("Swish(x)")
    plt.title("Swish Activation Function with Different Beta Values")
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
