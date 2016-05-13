#Linear Regression Learning(Multi Variable Matrix + b)

import tensorflow as tf

x_data = [[1.0, 2.0, 3.0, 4.0, 5.0],
          [2.0, 4.0, 6.0, 8.0, 10.0] ]
y_data  = [4.0, 8.0, 12.0, 16.0, 20.0]

W = tf.Variable(tf.random_uniform([1,2], -10.0, 10.0))
b = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = tf.matmul(W, X) + b

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
    #sess.run(train)
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0:
        print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W), sess.run(b)

print 'Answer'
print '[0.0, 6.0]', sess.run(hypothesis, feed_dict={X:[[0.0], [6.0]]})
print '[7.0, 0.0]', sess.run(hypothesis, feed_dict={X:[[7.0], [0.0]]})
print '[8.0, 8.0]', sess.run(hypothesis, feed_dict={X:[[8.0], [8.0]]})
