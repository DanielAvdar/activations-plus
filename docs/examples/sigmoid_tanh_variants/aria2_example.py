import matplotlib.pyplot as plt
import torch

from activations_plus.simple import aria2

x = torch.linspace(-5, 5, 200)

# Default parameters (alpha=1.5, beta=0.5)
y_default = aria2(x)

# Different alpha values
y_alpha_1 = aria2(x, alpha=1.0, beta=0.5)
y_alpha_2 = aria2(x, alpha=2.0, beta=0.5)

# Different beta values
y_beta_1 = aria2(x, alpha=1.5, beta=0.2)
y_beta_2 = aria2(x, alpha=1.5, beta=1.0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot different alpha values
ax1.plot(x.numpy(), y_default.numpy(), label="Default (α=1.5, β=0.5)")
ax1.plot(x.numpy(), y_alpha_1.numpy(), label="α=1.0, β=0.5")
ax1.plot(x.numpy(), y_alpha_2.numpy(), label="α=2.0, β=0.5")
ax1.set_title("ARiA2 with Different Alpha Values")
ax1.set_xlabel("Input")
ax1.set_ylabel("Output")
ax1.grid(alpha=0.3)
ax1.legend()

# Plot different beta values
ax2.plot(x.numpy(), y_default.numpy(), label="Default (α=1.5, β=0.5)")
ax2.plot(x.numpy(), y_beta_1.numpy(), label="α=1.5, β=0.2")
ax2.plot(x.numpy(), y_beta_2.numpy(), label="α=1.5, β=1.0")
ax2.set_title("ARiA2 with Different Beta Values")
ax2.set_xlabel("Input")
ax2.set_ylabel("Output")
ax2.grid(alpha=0.3)
ax2.legend()

plt.tight_layout()
fig.show()
