#Logistic(Binary, sigmoid) Classification

import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

print 'x', x_data
print 'y', y_data

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1,len(x_data)], -1.0, 1.0))

h = tf.matmul(W, X)     #matrix multiply
hypothesis = tf.div(1.0, 1.0 + tf.exp(-h))  #sigmoid(0.0 ~ 1.0)

#Cost Function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))
#cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis)))

#Minimize
a = tf.Variable(0.1)    #Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print 'Learning'
for step in range(1, 4001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 40 == 0:
        print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(hypothesis, feed_dict={X:x_data})

print 'Answer'
print sess.run(hypothesis, feed_dict={X:[[1], [2], [2]]}) > 0.5
print sess.run(hypothesis, feed_dict={X:[[1], [5], [5]]}) > 0.5
print sess.run(hypothesis, feed_dict={X:[[1], [8], [3]]}) > 0.5

print sess.run(hypothesis, feed_dict={X:[[1,1], [4,3], [3,5]]}) > 0.5


