import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

fig = plt.figure()

x = [1, 2, 3, 4, 5, 6, 7]
y = [1, 3, 4, 5, 8, 6, 2]

left, boottom, width, height = 0.1, 0.1, 0.8, 0.8
ax1 = fig.add_axes([left, boottom, width, height])
ax1.plot(x, y, 'r')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('ax1')

left, boottom, width, height = 0.2, 0.6, 0.25, 0.25
ax2 = fig.add_axes([left, boottom, width, height])
ax2.plot(x, y, 'b')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('ax2')
"""
left, boottom, width, height = 0.5, 0.2, 0.25, 0.25
ax2 = fig.add_axes([left, boottom, width, height])
ax2.plot(y, x, 'k')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('ax2')
"""
left, boottom, width, height = 0.5, 0.2, 0.25, 0.25
plt.axes([left, boottom, width, height])
plt.plot(y, x, 'k')
plt.xlabel('x')
plt.ylabel('y')
plt.title('ax2')
plt.show()


