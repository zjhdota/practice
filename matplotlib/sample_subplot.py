import matplotlib.pyplot as plt
import numpy as np

plt.figure()


# 分成2行2列，放在第一个位置
plt.subplot(2, 1, 1)
plt.plot([0,1], [0,1])

plt.subplot(2, 3, 4)
plt.plot([0,1], [0,2])

plt.subplot(2, 3, 5)
plt.plot([0,1], [0,3])

plt.subplot(2, 3, 6)
plt.plot([0,1], [0,4])
plt.show()
