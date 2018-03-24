# -*- encoding=utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 导入训练集
mnist = input_data.read_data_sets(r"MNIST_data", one_hot=True)

tf.set_random_seed(1)

learning_rate = 0.01 # 学习速率
train_steps = 2000 # 训练总次数
batch_size = 50 # 样本大小
n_inputs = 28 # 28 列
n_steps = 28 # 28 行
n_hidden_units = 128 # 神经元
n_classes = 10 # 输出结果分类0~9

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
    # shape = (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # shape = (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases = {
    # shape = (128,)
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units])),
    # shape = (10,)
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes]))
}

# 三部分组成input_layer, output_layer, cell
def RNN(X, weights, biases):
    # input_layer
    # -1 = 50 batch_size * 28 n_steps
    X = tf.reshape(x, [-1, n_inputs])
    # X_in.shape(50 batch_size * 28 n_steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in.shape(50 batch_size, 28 n_inputs, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    # cell
    # forget_bias = 1 不忘记
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=X_in, initial_state=init_state, time_major=False)
    # output_layer
    results = tf.matmul(final_state[1], weights['out']) + biases['out']
    return results

prediction = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    for i in range(train_steps):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        session.run([train_op], feed_dict={x: batch_xs, y:batch_ys})
        if i % 50 == 0:
            print(session.run(accuracy, feed_dict={x: batch_xs, y:batch_ys}))












