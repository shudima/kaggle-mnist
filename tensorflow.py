
# coding: utf-8

# In[1]:

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
get_ipython().magic(u'matplotlib inline')

from __future__ import division
import matplotlib.pyplot as plt
import tensorflow as tf


# In[2]:

epochs = 100000
batch_size = 100
use_dropout = False
dropout_keep_prob = 0.5
learning_rate = 1
hidden_layer = True
hidden_layer_neurons = 50
reg_rate = 0.0001

x = tf.placeholder(tf.float32, [None, 784])

if hidden_layer:
    W1 = tf.Variable(tf.random_normal([784, hidden_layer_neurons]))
    b1 = tf.Variable(tf.zeros([hidden_layer_neurons]))


    z1 = tf.matmul(x, W1) + b1
    a1 = tf.nn.sigmoid(z1)

    keep_prob = tf.placeholder(tf.float32)

    if use_dropout:
        a1 = tf.nn.dropout(a1, keep_prob)

    W2 = tf.Variable(tf.random_normal([hidden_layer_neurons, 10]))
    b2 = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(a1, W2) + b2)
else:
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    
    y = tf.nn.softmax(tf.matmul(x, W) + b)


y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


regularizers = (tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1) +
              tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2))

cross_entropy += reg_rate * regularizers

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
acc = []
train_acc = []

for i in range(epochs):
  print '\r' + str((i/epochs)*100) + '%',

  batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob : dropout_keep_prob})

  if i % 100 == 0:
      correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      curr_acc = sess.run(accuracy, feed_dict={x: mnist.test.images[:1000], y_: mnist.test.labels[:1000], keep_prob :dropout_keep_prob})  
      curr_train_acc = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys, keep_prob : dropout_keep_prob})     
      acc.append(curr_acc)
      train_acc.append(curr_train_acc)
    
    
plt.plot(acc)
plt.plot(train_acc, color='r')

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print ''
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob : 0.5}))


# In[12]:

import pandas as pd
import numpy as np


# In[34]:

df = pd.read_csv('test.csv')


# In[36]:

predicted = sess.run(y, feed_dict={x: df.values, keep_prob : dropout_keep_prob})
predicted = np.array([np.argmax(p) for p in predicted])

# In[39]:

df = df[['label']]
df['imageId'] = df.index + 1


# In[43]:

df.to_csv('tensorflow-sub.csv', index=False)


