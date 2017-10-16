import numpy as np
import matplotlib.pyplot as plt
import logistic_classifier as lc

# Find softmax
scores = [3.0, 1.0, 0.2]
print(lc.softmax(scores))

# Plot softmax curves
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
plt.plot(x, lc.softmax(scores).T, linewidth=2)
plt.show()
