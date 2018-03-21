import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function=None):
    # random_normal 随机正态分布
    with tf.name_scope('layer'):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

#  x 为 300行 1个特性
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
# 设置噪音 方差0.05
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

l1 = add_layer(x_data, 1, 10, tf.nn.relu)
prediction = add_layer(l1, 10, 1)

# reduction_indicees = [1] 按行求和
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_data - prediction), reduction_indices = [1]))
tf.summary.scalar('loss', loss)
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

init = tf.global_variables_initializer()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

with tf.Session() as session:

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/", session.graph)
    session.run(init)

    for i in range(1000):
        session.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            print('step=', i, 'loss=', session.run(loss, feed_dict={xs:x_data, ys:y_data}))
            result = session.run(merged, feed_dict={xs:x_data, ys:y_data})
            writer.add_summary(result, i)
            try:
                # 删除线
                ax.lines.remove(lines[0])
            except:
                pass
            prediction_value = session.run(prediction, feed_dict={xs: x_data})
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            plt.pause(0.1)




