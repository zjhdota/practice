import matplotlib.pyplot as plt
import numpy as np

a = np.random.rand(3, 3)

print(a)

# orign : 正反显示
#
plt.imshow(a, interpolation='nearest', cmap=plt.cm.bone, origin='lower')

# 压缩0.9
plt.colorbar(shrink=0.9)


plt.xticks(())
plt.yticks(())
plt.show()
