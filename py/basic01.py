#Tensor Flow Basic

import tensorflow as tf

sess = tf.Session()

a = tf.constant(2)
b = tf.constant(6)

#c = 2 + 6  #error
c = a + b

print "a + b: %i" % sess.run(c)
print "a * b: %i" % sess.run(a * b)
