import tensorflow as tf
a = tf.constant(1)
b = tf.constant(2)
with tf.Session() as session:
     print(session.run(a+b))

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
add = tf.add(a,b)
mul = tf.multiply(a,b)
tf.add
with tf.Session() as session:
    print(session.run(add, feed_dict={a:2,b:3}))
    print(session.run(mul, feed_dict={a:2,b:3}))

