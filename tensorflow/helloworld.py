# -*- encoding=utf-8 -*-
import tensorflow as tf
hello = tf.constant('hello world! by TensorFlow!')
session = tf.Session()
print(session.run(hello))
