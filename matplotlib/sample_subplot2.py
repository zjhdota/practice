import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

# method 1:
plt.figure()

ax1 = plt.subplot2grid((3, 3), (0,0), colspan=3, rowspan=1)
ax2 = plt.subplot2grid((3, 3), (1,0), colspan=2, rowspan=1)
ax3 = plt.subplot2grid((3, 3), (1,2), colspan=1, rowspan=2)
ax4 = plt.subplot2grid((3, 3), (2,0), colspan=1, rowspan=1)
ax5 = plt.subplot2grid((3, 3), (2,1), colspan=1, rowspan=1)

ax1.plot([1,2], [1,2])
ax2.plot([1,2], [1,2])
ax3.plot([1,2], [1,2])
ax4.plot([1,2], [1,2])
ax5.plot([1,2], [1,2])

ax1.set_title('ax1_title')

# method 2:
plt.figure()
gs = gridspec.GridSpec(3, 3)

ax1 = plt.subplot(gs[0, :])
ax2 = plt.subplot(gs[1, :2])
ax3 = plt.subplot(gs[1:, 2])
ax4 = plt.subplot(gs[-1, 0])
ax5 = plt.subplot(gs[-1, -2])

# method 3:

plt.figure()

# sahrexs 共享X轴
f, ((ax11),(ax12),(ax21, ax22)) = plt.subplots(2, 2, sharex=True, sharey=True)
ax11.scatter([1,2], [1,2])

plt.tight_layout()
plt.show()

