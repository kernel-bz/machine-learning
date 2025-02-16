'''
HelloWorld example using TensorFlow library.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import os;
os.environ['TF_CPP_MIN_LOG_LEVEL']='2';

import tensorflow as tf;

#Simple hello world using TensorFlow

# Create a Constant op
# The op is added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
hello = tf.constant('Hello, TensorFlow!');

# Start tf session
sess = tf.Session();

print (sess.run(hello));
