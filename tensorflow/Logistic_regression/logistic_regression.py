# -*- encoding=utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 导入训练集
mnist = input_data.read_data_sets(r"MNIST_data", one_hot=True)


# 学习速率
learnin_rate = 0.1
# 循环多少次
trainin_epochs = 25
#每次多少个数据
batch_size = 100
# 每隔多少个展示一次
display_step = 1

# 设置两个占位符None:表示任意长度,784个维度
X = tf.placeholder(tf.float32, [None, 784])
# 图片的实际标签0~9
Y = tf.placeholder(tf.float32, [None, 10])

# 初始化权重和偏移量为0
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 创建线性模型
# softmax激励(activation)函数或者链接(link)函数
# y = W * x + b
pred = tf.nn.softmax(tf.matmul(X,W) + b)

"""自己实现交叉熵函数
# 成本函数 交叉熵(cost-entropy)
# cross_entropy 是预测概率分布, Y实际概率分布
cross_entropy = -tf.reduce_sum(Y * tf.log(pred))
cross_entropy = tf.reduce_mean(cross_entropy, reduction_indices=1)
"""
"""
# 使用自带的交叉熵函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=pred))
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(pred), reduction_indices=1))
# 梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learnin_rate).minimize(cross_entropy)

# 初始化变量
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)

    for epoch in range(trainin_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            session.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
            cost = session.run(cross_entropy, feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += cost / total_batch
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    # 检测预测和真实标签的匹配程度
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    # 转换布尔值为浮点数，并取平均
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 计算模型在测试数据集上的正确率
    print(mnist.test.images, mnist.test.labels)
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
"""

# 调高正确率，构建一个卷积神经网络
# 权重初始化
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# 卷积和池化
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(X, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 密集连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropcut
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(Y*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                X:batch[0], Y: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={X: batch[0], Y: batch[1], keep_prob: 0.5})

    print("test accuracy %g"%accuracy.eval(feed_dict={
        X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0}))
