#Multi Classification(Softmax Classifier)

import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])

print 'x', x_data
print 'y', y_data

X = tf.placeholder("float", [None,3]) #x1, x2 and 1 (for bias)
Y = tf.placeholder("float", [None,3]) #A,B,C (for classes)

W = tf.Variable(tf.zeros([3,3]))

#hypothesis = tf.nn.softmax(tf.matmul(W, X))  #Softmax
hypothesis = tf.nn.softmax(tf.matmul(X, W))  #Softmax
#h = tf.matmul(X, W)     #matrix multiply
#hypothesis = tf.div(1.0, 1.0 + tf.exp(-h))

#Cross-Entropy Cost Function
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1))
#cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))

#Minimize
#a = tf.Variable(0.1)    #Learning rate, alpha
learning_rate = 0.001   #Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print 'Learning'

#with tf.Session() as sess:
#    sess.run(init)

for step in range(1, 2001):
    sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
    if step % 40 == 0:
        print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W)

print 'Answer'
a = sess.run(hypothesis, feed_dict={X:[[1, 11, 7]]})
print a, sess.run(tf.arg_max(a,1))

b = sess.run(hypothesis, feed_dict={X:[[1, 3, 4]]})
print a, sess.run(tf.arg_max(b,1))

c = sess.run(hypothesis, feed_dict={X:[[1, 1, 0]]})
print a, sess.run(tf.arg_max(c,1))

all = sess.run(hypothesis, feed_dict={X:[[1, 11, 7], [1, 3, 4], [1, 1, 0]]})
print all, sess.run(tf.arg_max(all,1))
