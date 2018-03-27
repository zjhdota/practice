import theano
import numpy as np
import theano.tensor as T
#import matplotlib.pyplot as plt


class Layer(object):
    def __init__(self, inputs, in_size, out_size, activation_function=None):
        self.W = theano.shared(np.random.normal(0, 1, (in_size, out_size)))
        self.bias = theano.shared(np.zeros((out_size,)) + 0.1)
        self.Wx_plus_b = T.dot(inputs, self.W) + self.bias
        self.activation_function = activation_function
        if self.activation_function is None:
            self.outputs = self.Wx_plus_b
        else:
            self.outputs = self.activation_function(self.Wx_plus_b)

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

"""
plt.scatter(x, y)
plt.show()
"""

# 定义变量类型
x = T.dmatrix('x')
y = T.dmatrix('y')

# 定义隐藏层
l1 = Layer(x, 1, 10, T.nnet.relu)
l2 = Layer(l1.outputs, 10, 1, None)

# 计算cost
cost = T.mean(T.square(l2.outputs - y))

# 计算梯度
gW1, gb1, gW2, gb2 = T.grad(cost, [l1.W, l1.bias, l2.W, l2.bias])

# 进行梯度下降
learning_rate = 0.05
train = theano.function(
    inputs=[x, y],
    outputs=cost,
    updates=[(l1.W, l1.W - learning_rate * gW1),
             (l1.bias, l1.bias - learning_rate * gb1),
             (l2.W, l2.W - learning_rate * gW2),
             (l2.bias, l2.bias - learning_rate * gb2)])
predict = theano.function(inputs=[x], outputs=l2.outputs)

for i in range(1000):
    err = train(x_data, y_data)
    if i % 50 == 0:
        print(err)



























