import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1, 1, 50)
y1 = 2*x + 1

y2 = x**2

# 生成窗口来显示图像
plt.figure()
plt.plot(x, y1)

# 生成新的窗口来显示图像
plt.figure(num=3, figsize=(4,4))
l1, = plt.plot(x, y2, label="y=2*x+1")
l2, = plt.plot(x, y1, color='r', linewidth=1.0, linestyle='--', label='y=x**2')

# 设置x轴显示(-1, 1), y轴显示(-1, 2)
plt.xlim((-1, 1))
plt.ylim((-1, 2))

# 设置X轴标签， 设置Y轴标签
plt.xlabel('x')
plt.ylabel('y')

# 生成新的ticks
new_ticks = np.linspace(-1, 2, 5)
print(new_ticks)
# 更改ticks
plt.xticks(new_ticks)
plt.yticks([-2, -1.8, -1, 1.22, 3],
           ['$really bad$', 'bad', 'normal', 'good', 'really good'])


#gca
ax = plt.gca()
# 去除上、右的黑边
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# 设置坐标轴上的数字显示的位置，bottom:显示在底部,默认是none
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# 修改坐标轴的位置
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

# 设置注解
x0 = 1
y0 = 2*x0 + 1
# s,size表示点的大小，默认20
plt.scatter(x0, y0, s=50, color='b')
plt.plot([x0, x0], [y0, 0], 'k--', linewidth=2.5)

# method 1:
plt.annotate(r'$2x+1=%s$' % y0, xy=(x0, y0), xycoords='data', xytext=(+30, -30), textcoords="offset points", fontsize=16, arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=.2'))

# method 2:
plt.text(-0, 1, r'$This\ is\ some\ text. \mu\ \sigma_i\ \alpha_t$', fontdict={'size':16, 'color':'r'})

# 设置ticks透明度
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(12)
    label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.7))

plt.legend(handles=[l1, l2,],labels=['y=2*x+1', 'y=x**2'], loc='best')

plt.show()
