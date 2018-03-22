import tensorflow as tf
import numpy as np

"""
# save
W = tf.Variable([[1, 2, 3], [3, 4, 5]], dtype=tf.float32, name='Weights')
b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name='biases')

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as session:
    session.run(init)
    save_path = saver.save(session,  'save_file/save_ckpt')
    print("Save to path", save_path)
"""

# restore variable
W = tf.Variable(np.arange(6).reshape((2,3)), dtype=tf.float32, name='Weights')
b = tf.Variable(np.arange(3).reshape((1,3)), dtype=tf.float32, name='biases')

# not need init step

saver = tf.train.Saver()

with tf.Session() as session:
    saver.restore(session, 'save_file/save_ckpt')
    print('Weights', session.run(W))
    print('biases', session.run(b))


