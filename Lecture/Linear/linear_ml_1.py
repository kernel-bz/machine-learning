#Linear Regression Learning(W=1)

import tensorflow as tf

x_data = [1., 2., 3.]
y_data = [3., 6., 9.]

W = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X

#Cost Function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#Minimize(gradient descent algorithm)
descent = W - tf.mul(0.1, tf.reduce_mean(tf.mul((tf.mul(W, X) - Y), X)))
#descent = W - 0.1 x mean((wx - y)x)
update = W.assign(descent)
#W = W - w

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print 'first W = ', sess.run(W)

print 'Learning'
for step in range(1, 100):
    sess.run(update, feed_dict={X:x_data, Y:y_data})
    print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W)

print 'Answer'
print ' 5.0', sess.run(hypothesis, feed_dict={X:5.0})
print '10.0', sess.run(hypothesis, feed_dict={X:10.0})
