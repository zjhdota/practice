import numpy as np

a = np.array([[1,2,3],
         [4,5,6]],
         dtype=np.int)

print(a.ndim)
print(a.shape)
print(a.dtype)

# 生成0矩阵
a = np.zeros((3,4), dtype=np.int16)
print(a)

# 生成随机数矩阵
a = np.empty((2,3))
print(a)

# 类似range
a = np.arange(12).reshape((3, 4))
print(a)

# 生成几段
a = np.linspace(1, 10, 6).reshape((2,3))
print(a)

a = np.array([10, 20, 30, 40])
b = np.arange(4)
print(a-b)
print(a+b)
print(a*b)

b = b ** 2
print(b)
print(b==0)

# 矩阵相乘
c = np.dot(a, b)
print(c)

# 生成0~1的数字
a = np.random.random((2,4))
print(a)

# axis=1 对行运算；axis=0 对列运算
print(np.sum(a, axis=1))
print(np.max(a))
print(np.min(a))


a = np.arange(2, 14).reshape((3,4))
print(np.argmax(a))
print(np.argmin(a))

# 平均值
print(np.mean(a))
print(a.mean())

# 中位数
print(np.median(a))

# 元素累加 输出一个向量
print(np.cumsum(a))

# 元素差
print(np.diff(a))

# 排序
print(np.sort(a))

# 转置
print(np.transpose(a))
print(a.T)
# 变化小于3大于5的数
print(np.clip(a, 3, 5))

a = np.arange(3, 15).reshape((3,4))
print(a)

# 打印出所有数
print(a[:1, :])

# flatten() 合并矩阵
print(a.flatten())
for item in a.flat:
    print(item)


# 合并矩阵
a = np.array([1, 1, 1])
b = np.array([2, 2, 2])

# 上下合并
c = np.vstack((a,b))
print(c)
print(c.shape)

# 左右合并
d = np.hstack((a,b))
print(d)
print(d.shape)

# 横向数列变为纵向数列
print(a.reshape(a.size,1))
print(a[:, np.newaxis])
a = a[:, np.newaxis]
b = b[:, np.newaxis]
print(a, b)

#
c = np.concatenate((a,b,b), axis=1)
print(c)

#numpy 的分割
a = np.arange(12).reshape((3,4))
print(a)

# 横向分割
c = np.split(a, 3, axis=0)
print(c)

# 纵向分割
c = np.split(a, 2, axis=1)
print(c)

# 不等分割
c = np.split(a, [1,1,2], axis=1)
print(c)

c = np.array_split(a, 3, axis=1)
print(c)

c = np.hsplit(a, [1,1,2])
print(c)

#from copy import deepcopy,copy

# numpy的赋值
a = np.array([0, 1, 2, 3])
b = a
c = a
d = a
a[0] = 4

print(a)
print(b)
print(c)
print(d)

# deepcopy
b = a.copy()
a[0] = 5
print(a)
print(b)

















