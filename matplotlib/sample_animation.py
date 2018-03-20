import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

fig, ax = plt.subplots()

x = np.arange(0, 2* np.pi, 0.01)
line, = ax.plot(x, np.sin(x))

def ani(i):
    line.set_ydata(np.sin(x+i/10))
    return line,


def init():
    line.set_ydata(np.sin(x))
    return line,

# interval 频率, frames: 帧数，blit：是否全部更新， init_func:初始化图像
ani = animation.FuncAnimation(fig=fig, func=ani, frames=100, init_func=init, interval=20, blit=True)

plt.show()


