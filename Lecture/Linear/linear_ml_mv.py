#Linear Regression Learning(Multi Variable)

import tensorflow as tf

#x1_data = [1.0, 0.0, 3.0, 0.0, 5.0]
#x2_data = [0.0, 2.0, 0.0, 4.0, 0.0]
x1_data = [1.0, 2.0, 3.0, 4.0, 5.0]
x2_data = [1.0, 2.0, 3.0, 4.0, 5.0]
y_data  = [1.0, 2.0, 3.0, 4.0, 5.0]

W1 = tf.Variable(tf.random_uniform([1], -10.0, 10.0))
W2 = tf.Variable(tf.random_uniform([1], -10.0, 10.0))
b = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

X1 = tf.placeholder(tf.float32)
X2 = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W1*X1 + W2*X2 + b

#Cost Function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#Minimize(gradient descent algorithm)
#descent = W - tf.mul(0.1, tf.reduce_mean(tf.mul((tf.mul(W, X) - Y), X)))
#update = W.assign(descent)
a = tf.Variable(0.01)    #Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print 'Learning'
for step in range(1, 801):
    #sess.run(update, feed_dict={X:x_data, Y:y_data})
    sess.run(train, feed_dict={X1:x1_data, X2:x2_data, Y:y_data})
    if step % 20 == 0:
        print step, sess.run(cost, feed_dict={X1:x1_data, X2:x2_data, Y:y_data}), sess.run(W1), sess.run(W2), sess.run(b)

print 'Answer'
print ' 0.0, 6.0', sess.run(hypothesis, feed_dict={X1:0.0, X2:6.0})
print ' 7.0, 0.0', sess.run(hypothesis, feed_dict={X1:7.0, X2:0.0})
print ' 8.0, 8.0', sess.run(hypothesis, feed_dict={X1:8.0, X2:8.0})
