import tensorflow as tf
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

def add_layer(inputs, in_size, out_size, activation_function=None):
    # random_normal 随机正态分布
    with tf.name_scope('layer'):
        Weights = tf.Variable(tf.random_normal([in_size, out_size], dtype=tf.float64))
        biases = tf.Variable(tf.zeros([1, out_size], dtype=tf.float64) + 0.1)
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

xs = tf.placeholder(dtype=tf.float64, shape=[None, 64]) # 8x8
ys = tf.placeholder(dtype=tf.float64, shape=[None, 10])
keep_prob = tf.placeholder(dtype=tf.float64)

l1 = add_layer(xs, 64, 100, tf.nn.tanh)
prediction = add_layer(l1, 100, 10, tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))

tf.summary.scalar('cross_entropy', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.6).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as session:

    train_writer = tf.summary.FileWriter('logs/train', session.graph)
    test_writer = tf.summary.FileWriter('logs/test', session.graph)
    merged = tf.summary.merge_all()

    session.run(init)
    for i in range(500):

        session.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.5})
        if i % 50 == 0:
            print('step=', i, 'loss=', session.run(cross_entropy, feed_dict={xs:X_train, ys:y_train, keep_prob: 0.5}))
            train_result = session.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
            test_result = session.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
            train_writer.add_summary(train_result, i)
            test_writer.add_summary(test_result, i)





















