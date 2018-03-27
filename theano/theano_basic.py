# -*- encoding=utf-8 -*-
import theano
import numpy as np
import theano.tensor as T
from theano import function

# 相加
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y

f = function([x, y], z)

print(f(2, 3))

# 输出公式
from theano import pp

print(pp(z))

# 矩阵相加
x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f = function([x, y], z)
print(f(np.arange(12).reshape(3,4), 10*np.ones((3,4))))

# activation function : s
x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x)) # logistic or soft step ,sigmoid
logistic = theano.function([x], s)
print(logistic([[0, 1],[2, 3]]))

# 定义多个矩阵，并输出多值
a, b = T.dmatrices('a','b')
diff = a - b
abs_diff =abs(diff)
diff_squared = diff ** 2
f = function([a, b], [diff, abs_diff, diff_squared])

print(f(np.ones((2,2)), np.arange(4).reshape(2,2)))

# 设置参数默认值，名字
w = T.dscalar('w')
x, y, w = T.dscalars('x', 'y', 'w')
z = (x+y) * w
f = function([x, theano.In(y, value=1), theano.In(w, value=2, name='weight')], z)

print(f(23, ))


# shared变量
state = theano.shared(np.array(0, dtype=np.float64), 'state')
inc = T.scalar('inc', dtype=state.dtype)
accumulator = function([inc], state, updates=[(state, state+inc)])

print(state.get_value())
accumulator(1)
print(state.get_value())
accumulator(10)
print(state.get_value())

state.set_value(1)
print(state.get_value())

# 定义暂时shared
tmp_func = state * 2 + inc
a = T.scalar('a', dtype=state.dtype)
skip_shared = function([inc, a], tmp_func, givens=[(state, a)])
print(skip_shared(2, 3))
print(state.get_value())









