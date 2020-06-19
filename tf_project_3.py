'''
an example:
    solving a linear regression:
    model: N=1000
           D=3
           x = np.random.random((N,D))
           w = np.random.random((D,1))
           y = x@w + np.random.randn(N,1)*0.2
'''

import tensorflow as tf
import numpy as np
N = 1000
D = 3
x = np.random.random((N,D))
w = np.random.random((D, 1))
y = x@w + np.random.randn(N,1)*0.2

tf.reset_default_graph()
features = tf.placeholder(tf.float32, shape=(None, D))
target = tf.placeholder(tf.float32, shape=(None, 1))

weights = tf.get_variable("w", shape=(D,1), dtype=tf.float32)
predictions = features@weights

loss = tf.reduce_mean((target - predictions)**2)
optimizer = tf.train.GradientDescentOptimizer(0.1)
step = optimizer.minimize(loss)

# put loss in summary on order to view it in tensorboard
tf.summary.scalar('curr_loss', loss)
summaries = tf.summary.merge_all()

s = tf.InteractiveSession()
summary_writer = tf.compat.v1.summary.FileWriter('./logs/project_3', s.graph)
saver = tf.compat.v1.train.Saver(tf.compat.v1.trainable_variables())
s.run(tf.compat.v1.global_variables_initializer())
for i in range(10000000):
    # _, curr_loss, curr_weight = s.run([step, loss, weights], feed_dict={features:x, target:y})
    _, curr_summary, curr_weight = s.run([step, summaries, weights], feed_dict={features: x, target: y})
    summary_writer.add_summary(curr_summary, i)
    summary_writer.flush()
    if i%100==0:
        saver.save(s, "./logs/project_3/model.ckpt", global_step=i)
        print(i, curr_summary)
        # print()


