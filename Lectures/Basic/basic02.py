#Tensor Flow Basic

import os;
os.environ['TF_CPP_MIN_LOG_LEVEL']='2';

import tensorflow as tf;

sess = tf.Session();

a = tf.placeholder(tf.int16);
b = tf.placeholder(tf.int16);

#define operation
add = tf.add(a, b);
mul = tf.multiply(a, b);

#with tf.Session() as sess: <tab>
print ("add: %i" % sess.run(add, feed_dict={a:2, b:6}));
print ("mul: %i" % sess.run(mul, feed_dict={a:2, b:6}));
