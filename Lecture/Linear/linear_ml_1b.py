#Linear Regression Learning(W=1, b)

import tensorflow as tf

x_data = [1.0, 2.0, 3.0]
y_data = [1.0, 2.0, 3.0]

W = tf.Variable(tf.random_uniform([1], -10.0, 10.0))
b = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X + b

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
    #sess.run(update, feed_dict={X:x_data, Y:y_data})
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0:
        print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W), sess.run(b)

print 'Answer'
print ' 5.0', sess.run(hypothesis, feed_dict={X:5.0})
print '10.0', sess.run(hypothesis, feed_dict={X:10.0})
