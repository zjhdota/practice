import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# 导入训练集
mnist = input_data.read_data_sets(r"MNIST_data", one_hot=True)

############### type 1 ###############
learning_rate = 0.01
train_steps = 1000
batch_size = 50
display_step = 50
examples_to_show = 10

# img shape: 28 * 28
n_input = 784
n_hidden_1 = 256
n_hidden_2 = 128

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
}

biases = {
    'encoder_h1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_input]))
}

def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_h1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_h2']))
    return layer_2

def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_h1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_h2']))
    return layer_2

X = tf.placeholder(tf.float32, [None, 784])

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

prediction = decoder_op
y_true = X

# 最小二乘法
cost = tf.reduce_mean(tf.pow((y_true - prediction), 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as session:

    session.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    for i in range(5):
        for epoch in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = session.run([optimizer, cost], feed_dict={X: batch_xs})
            if epoch % display_step == 0:
                print('epoch=', epoch,'cost=', c)

    print("Optimization Finished!")

    # # Applying encode and decode over test set
    encode_decode = session.run(
        prediction, feed_dict={X: mnist.test.images[:examples_to_show]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    plt.show()

"""
############### type 2 ###############
learning_rate = 0.01
train_steps = 1000
batch_size = 50
display_step = 50
examples_to_show = 10

n_input = 784
n_hidden_1 = 128
n_hidden_2 = 64
n_hidden_3 = 8
n_hidden_4 = 2

weights = {
    'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],)),
    'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],)),
    'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],)),
    'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4],)),

    'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_3],)),
    'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2],)),
    'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1],)),
    'decoder_h4': tf.Variable(tf.truncated_normal([n_hidden_1, n_input],)),
    }
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),

    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b4': tf.Variable(tf.random_normal([n_input])),
    }

def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    layer_4 = tf.add(tf.matmul(layer_3, weights['encoder_h4']),
                                    biases['encoder_b4'])
    return layer_4


def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                biases['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
                                biases['decoder_b4']))
    return layer_4

X = tf.placeholder(tf.float32, [None, 784])

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

prediction = decoder_op
y_true = X

# 最小二乘法
cost = tf.reduce_mean(tf.pow((y_true - prediction), 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as session:

    session.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    for i in range(5):
        for epoch in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = session.run([optimizer, cost], feed_dict={X: batch_xs})
            if epoch % display_step == 0:
                print('epoch=', epoch,'cost=', c)

    print("Optimization Finished!")

    # # Applying encode and decode over test set
    encode_decode = session.run(
        prediction, feed_dict={X: mnist.test.images[:examples_to_show]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    plt.show()
"""
"""
encoder_result = session.run(encoder_op, feed_dict={X: mnist.test.images[:2000]})
plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=mnist.test.labels.reshape(-1,)[:2000])
plt.colorbar()
plt.show()
"""

