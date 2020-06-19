"""
2. Optimize
use tensorboard to view your training procudure
    call:  $: tensorflow --logdir=./logs
    open:  http://127.0.0.1:6006
"""

import tensorflow as tf
import numpy as np

tf.reset_default_graph()
x = tf.get_variable("x", shape=(), dtype=tf.float32)
f = x**2

tf.summary.scalar('curr_x', x)
tf.summary.scalar('curr_f', f)
summaries = tf.summary.merge_all()


optimizer = tf.train.GradientDescentOptimizer(0.1)
step = optimizer.minimize(f, var_list=[x])
s = tf.InteractiveSession()
summary_writer = tf.compat.v1.summary.FileWriter("./logs/project_2", s.graph)
s.run(tf.compat.v1.global_variables_initializer())
for i in range(1000000):
    _, curr_summaries = s.run([step, summaries])
    summary_writer.add_summary(curr_summaries, i)
    summary_writer.flush()
    if i%100 == 0:
        print(i)

