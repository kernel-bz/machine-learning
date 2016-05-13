#Linear Regression(W=0.5)

import tensorflow as tf

X = [1.0, 2.0, 3.0]
Y = [0.5, 1.0, 1.5]
m = n_samples = len(X)

W = tf.placeholder(tf.float32)

hypothesis = tf.mul(X, W)

cost = tf.reduce_sum(tf.pow(hypothesis - Y, 2))/(m)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(-20, 30):
    print i, i*0.1, sess.run(cost, feed_dict={W: i*0.1})
