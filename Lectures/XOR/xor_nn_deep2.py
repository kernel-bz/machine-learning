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
#Deep Neural Network
W1 = tf.Variable(tf.random_uniform([4, 8], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([8, 2], -1.0, 1.0))
W3 = tf.Variable(tf.random_uniform([2, 6], -1.0, 1.0))
W4 = tf.Variable(tf.random_uniform([6, 4], -1.0, 1.0))
W5 = tf.Variable(tf.random_uniform([4, 8], -1.0, 1.0))
W6 = tf.Variable(tf.random_uniform([8, 2], -1.0, 1.0))
W7 = tf.Variable(tf.random_uniform([2, 6], -1.0, 1.0))
W8 = tf.Variable(tf.random_uniform([6, 4], -1.0, 1.0))
W9 = tf.Variable(tf.random_uniform([4, 8], -1.0, 1.0))
W10 = tf.Variable(tf.random_uniform([8, 2], -1.0, 1.0))
W11 = tf.Variable(tf.random_uniform([2, 6], -1.0, 1.0))
W12 = tf.Variable(tf.random_uniform([6, 4], -1.0, 1.0))
W13 = tf.Variable(tf.random_uniform([4, 8], -1.0, 1.0))
W14 = tf.Variable(tf.random_uniform([8, 2], -1.0, 1.0))
W15 = tf.Variable(tf.random_uniform([2, 10], -1.0, 1.0))
W16 = tf.Variable(tf.random_uniform([10, 10], -1.0, 1.0))
W17 = tf.Variable(tf.random_uniform([10, 4], -1.0, 1.0))

b1 = tf.Variable(tf.zeros([8]), name="bias1")
b2 = tf.Variable(tf.zeros([2]), name="bias2")
b3 = tf.Variable(tf.zeros([6]), name="bias3")
b4 = tf.Variable(tf.zeros([4]), name="bias4")
b5 = tf.Variable(tf.zeros([8]), name="bias5")
b6 = tf.Variable(tf.zeros([2]), name="bias6")
b7 = tf.Variable(tf.zeros([6]), name="bias7")
b8 = tf.Variable(tf.zeros([4]), name="bias8")
b9 = tf.Variable(tf.zeros([8]), name="bias9")
b10 = tf.Variable(tf.zeros([2]), name="bias10")
b11 = tf.Variable(tf.zeros([6]), name="bias11")
b12 = tf.Variable(tf.zeros([4]), name="bias12")
b13 = tf.Variable(tf.zeros([8]), name="bias13")
b14 = tf.Variable(tf.zeros([2]), name="bias14")
b15 = tf.Variable(tf.zeros([10]), name="bias15")
b16 = tf.Variable(tf.zeros([10]), name="bias16")
b17 = tf.Variable(tf.zeros([4]), name="bias17")

#h = tf.matmul(W, X)     #matrix multiply
#hypothesis = tf.div(1.0, 1.0 + tf.exp(-h))
#Neural Network hypothesis
L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
L3 = tf.sigmoid(tf.matmul(L2, W2) + b2)
L4 = tf.sigmoid(tf.matmul(L3, W3) + b3)
L5 = tf.sigmoid(tf.matmul(L4, W4) + b4)
L6 = tf.sigmoid(tf.matmul(L5, W5) + b5)
L7 = tf.sigmoid(tf.matmul(L6, W6) + b6)
L8 = tf.sigmoid(tf.matmul(L7, W7) + b7)
L9 = tf.sigmoid(tf.matmul(L8, W8) + b8)
L10 = tf.sigmoid(tf.matmul(L9, W9) + b9)
L11 = tf.sigmoid(tf.matmul(L10, W10) + b10)
L12 = tf.sigmoid(tf.matmul(L11, W11) + b11)
L13 = tf.sigmoid(tf.matmul(L12, W12) + b12)
L14 = tf.sigmoid(tf.matmul(L13, W13) + b13)
L15 = tf.sigmoid(tf.matmul(L14, W14) + b14)
L16 = tf.sigmoid(tf.matmul(L15, W15) + b15)
L17 = tf.sigmoid(tf.matmul(L16, W16) + b16)
hypothesis = tf.sigmoid(tf.matmul(L17, W17) + b17)

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
            print step, sess.run(cost, feed_dict={X:x_data, Y:y_data})

    print 'Test Model'
    prediction = tf.equal(tf.floor(hypothesis+0.5), Y)
    #Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(prediction, "float"))
    print sess.run([hypothesis, tf.floor(hypothesis+0.5), prediction, accuracy], feed_dict={X:x_data, Y:y_data} )
    print "Accuracy:", accuracy.eval({X:x_data, Y:y_data})
