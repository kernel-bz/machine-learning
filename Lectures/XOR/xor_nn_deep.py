#XOR Neural Network with Logistic Classification
#Editted by JungJaeJoon(rgbi3307@nate.com) on the www.kernel.bz

import tensorflow as tf
import numpy as np

xy = np.loadtxt('xor_train.txt', unpack=True, dtype='float32')

#xy.transpose()

x_data = xy[0:-1]
y_data = xy[-1]

print 'x', x_data
print 'y', y_data

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#W = tf.Variable(tf.random_uniform([1,len(x_data)], -1.0, 1.0))
#Neural Network
W1 = tf.Variable(tf.random_uniform([4, 8], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([8, 2], -1.0, 1.0))
W3 = tf.Variable(tf.random_uniform([2, 6], -1.0, 1.0))
W4 = tf.Variable(tf.random_uniform([6, 4], -1.0, 1.0))

b1 = tf.Variable(tf.zeros([8]), name="bias1")
b2 = tf.Variable(tf.zeros([2]), name="bias2")
b3 = tf.Variable(tf.zeros([6]), name="bias3")
b4 = tf.Variable(tf.zeros([4]), name="bias4")

#h = tf.matmul(W, X)     #matrix multiply
#hypothesis = tf.div(1.0, 1.0 + tf.exp(-h))
#Neural Network hypothesis
L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
L3 = tf.sigmoid(tf.matmul(L2, W2) + b2)
L4 = tf.sigmoid(tf.matmul(L3, W3) + b3)
hypothesis = tf.sigmoid(tf.matmul(L4, W4) + b4)

#Cost Function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))

#Minimize
a = tf.Variable(0.1)    #Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    print 'Learning'
    for step in range(1, 1001):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step % 100 == 0:
            print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W1), sess.run(W2), sess.run(W3), sess.run(W4)

    print 'Test Model'
    prediction = tf.equal(tf.floor(hypothesis+0.5), Y)
    #Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(prediction, "float"))
    print sess.run([hypothesis, tf.floor(hypothesis+0.5), prediction, accuracy], feed_dict={X:x_data, Y:y_data} )
    print "Accuracy:", accuracy.eval({X:x_data, Y:y_data})
