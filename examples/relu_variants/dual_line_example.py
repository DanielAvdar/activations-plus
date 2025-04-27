import matplotlib.pyplot as plt
import torch

from activations_plus.simple.relu_variants import dual_line

x = torch.linspace(-3, 3, 200)
y = dual_line(x, a=1.0, b=0.01, m=0.0)
fig, ax = plt.subplots()
ax.plot(x.numpy(), y.numpy())
ax.set_title("Dual Line (a=1.0, b=0.01, m=0.0)")
ax.set_xlabel("Input")
ax.set_ylabel("Output")
ax.grid(alpha=0.3)
# fig.show()  # This will be mocked in tests
