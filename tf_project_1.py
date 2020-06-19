'''
1. first tf program
'''
import tensorflow as tf
import numpy as np

# definition
tf.reset_default_graph()
# a = tf.constant(np.ones((2,2), dtype=np.float32))
b = tf.Variable(tf.ones((2,2)))
c = tf.placeholder(np.float, (2,2))
d = b@c

s = tf.InteractiveSession()
s.run(tf.compat.v1.global_variables_initializer())
s.run(d, feed_dict = {c:np.ones((2,2))})

tf.print(d)