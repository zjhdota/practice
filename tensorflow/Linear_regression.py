# -*- encoding=utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
rng = np.random

# 学习速率
learning_rate = 0.1
# 训练次数
trainin_epochs = 8000
# 每隔多少显示次数
display_step = 50

#样本数据
# Training Data
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10
#train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
#train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])

# 数据x的总数
n_samples = train_X.shape[0]
print(n_samples)

# tf Graph Input
# 设定两个占位符
X = tf.placeholder('float')
Y = tf.placeholder('float')
# W 斜率
# b 截距
# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name='bias')
# Construct a linear model
activation = tf.add(tf.multiply(X,W),b)

# reduce_sum: n维求和
# reduce_mean: 求平均值
# tf.square : 求平方
# Minimize the squared errors
cost = tf.reduce_sum(tf.pow(Y-activation, 2))/(2 * n_samples)
cost = tf.reduce_mean(tf.square(Y-activation))/5000
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 初始化变量
# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as session:
    session.run(init)

    # Fit all training data
    # 通过优化器去优化
    for epoch in range(trainin_epochs):
        for (x, y) in zip(train_X, train_Y):
            session.run(optimizer, feed_dict={X:x,Y:y})

        #Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(session.run(cost, feed_dict={X: train_X, Y:train_Y})), \
                "W=", session.run(W), "b=", session.run(b))
    print("Optimization Finished!")
    print("cost=", session.run(cost, feed_dict={X: train_X, Y: train_Y}), \
          "W=", session.run(W), "b=", session.run(b))

    #Graphic display
    # 'ro' 红色的圈
    plt.plot(train_X, train_Y, 'ro', label="Original data")
    plt.plot(train_X, session.run(W) * train_X + session.run(b), label="Fitted line")
    plt.legend()
    plt.show()
