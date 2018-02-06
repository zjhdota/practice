# -*- encoding=utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 导入训练集
mnist = input_data.read_data_sets(r"C:\Users\zjhdo\Desktop\mnist测试集", one_hot=True)

# 学习速率
learnin_rate = 0.1
# 循环多少次
trainin_epochs = 20
#
batch_size = 100
# 每隔多少个展示一次
display_step = 1

# 设置两个占位符None:表示任意长度,784个维度
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# sfotmax激励(activation)函数或者链接(link)函数
# y = W * x + b
pred = tf.nn.softmax(tf.matmul(X,W) + b)

# 成本函数 交叉熵(cost-entropy)
# cross_entropy 是预测概率分布, Y实际概率分布
cross_entropy = -tf.reduce_sum(Y * tf.log(pred))


cost = tf.reduce_mean(cross_entropy, reduction_indices=1)
cost = cross_entropy
# 梯度下降优化器
optimizer = tf.train.GradientdiDescentOptimizer(learnin_rate).minimize(cost)

# 初始化变量
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)

    for epoch in range(trainin_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = session.run([optimizer, cost], feed_dict={X:batch_xs,Y:batch_ys})
            avg_cost += c/total_batch
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
