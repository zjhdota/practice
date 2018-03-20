import matplotlib.pyplot as plt
import numpy as np

n = 1024
X = np.random.normal(0, 1, n)
Y = np.random.normal(0, 1, n)

T = np.arctan2(Y, X) # for color value
t = np.arange(1024)
plt.scatter(X, Y, s=75, c=t, alpha=0.7)

plt.xlim((-1.5, 1.5))
plt.ylim((-1.5, 1.5))

plt.figure()
plt.scatter(np.arange(5), np.arange(5))

plt.show()
