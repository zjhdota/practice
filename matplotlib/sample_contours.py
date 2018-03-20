import matplotlib.pyplot as plt
import numpy as np

def f(x, y):
    return (1 - x / 2 + x**5 + y**3) * np.exp(-x**2-y**2)

n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
X, Y = np.meshgrid(x, y)

# 分成8份
plt.contourf(X, Y, f(X, Y), 8, alpha=0.75, cmap=plt.cm.hot)

c = plt.contour(X, Y, f(X, Y), 8, colors='k', linewidths=(0.5))

plt.clabel(c, inline=True, fontsize=10)

# 去掉坐标轴
plt.xticks(())
plt.yticks(())

plt.show()
