#Tensor Flow Basic

import tensorflow as tf

sess = tf.Session()

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

#define operation
add = tf.add(a, b)
mul = tf.mul(a, b)

#with tf.Session() as sess: <tab>
print "add: %i" % sess.run(add, feed_dict={a:2, b:6})
print "mul: %i" % sess.run(mul, feed_dict={a:2, b:6})
