#Linear Regression Learning(Multi Variable in File)

import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

print 'x', x_data
print 'y', y_data

W = tf.Variable(tf.random_uniform([1,3], -10.0, 10.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#hypothesis = tf.matmul(W, X) + b
hypothesis = tf.matmul(W, X)

#Cost Function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#Minimize(gradient descent algorithm)
#descent = W - tf.mul(0.1, tf.reduce_mean(tf.mul((tf.mul(W, X) - Y), X)))
#update = W.assign(descent)
a = tf.Variable(0.1)    #Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print 'Learning'
for step in range(1, 801):
    #sess.run(train)
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0:
        #print step, sess.run(cost), sess.run(W)
        print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W)

print 'Answer'
print '[0.0, 6.0]', sess.run(hypothesis, feed_dict={X:[[1.0], [0.0], [6.0]]})
print '[7.0, 0.0]', sess.run(hypothesis, feed_dict={X:[[1.0], [7.0], [0.0]]})
print '[8.0, 8.0]', sess.run(hypothesis, feed_dict={X:[[1.0], [8.0], [8.0]]})
