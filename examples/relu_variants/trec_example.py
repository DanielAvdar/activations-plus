import matplotlib.pyplot as plt
import torch

from activations_plus.simple import trec

x = torch.linspace(-3, 3, 200)
y = trec(x, a=0.0)
fig, ax = plt.subplots()
ax.plot(x.numpy(), y.numpy())
ax.set_title("Truncated Rectified (a=0.0)")
ax.set_xlabel("Input")
ax.set_ylabel("Output")
ax.grid(alpha=0.3)
fig.show()  # This will be mocked in tests
