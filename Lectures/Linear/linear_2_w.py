#Linear Regression(W=2)
import os;
os.environ['TF_CPP_MIN_LOG_LEVEL']='2';

import tensorflow as tf

X = [1., 2., 3.]
Y = [2., 4., 6.]
m = n_samples = len(X)

W = tf.placeholder(tf.float32)

hypothesis = tf.multiply(X, W)

cost = tf.reduce_sum(tf.pow(hypothesis - Y, 2))/(m)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(-20, 30):
    print (i, i*0.1, sess.run(cost, feed_dict={W: i*0.1}))
