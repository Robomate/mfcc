# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 13:21:16 2016

@author: praktiku
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import numpy as np
import tensorflow as tf
import os, re
import datetime 
import matplotlib.pyplot as plt  
from PIL import Image
plt.close('all')

# remove warnings from tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

########################################################################
#init, define and train: CNN model
########################################################################

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#init: cost history
cost_history = np.empty(shape=[0],dtype=float)
#init: accuracy
acc_history = np.empty(shape=[0],dtype=float)
#init: batches correct
correctbatch = np.empty(shape=[0],dtype=float)


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
 
#1.conv layer                       
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#print h_conv1
h_pool1 = max_pool_2x2(h_conv1)

#2.conv layer   
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Densely Connected Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#Train and Evaluate the Model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

t1_1 = datetime.datetime.now()
#init all:
sess = tf.InteractiveSession()
#save and restore all the variables:
saver = tf.train.Saver()
#start session:
sess.run(tf.global_variables_initializer())

epochs = 5000  #5000: gpu(quadk620):3min/(i7,xeon):10min/(i5):43min
batsize_train=50
for i in range(epochs):
  batch = mnist.train.next_batch(batsize_train)
  print(batch[0].shape)
  print(batch[1].shape)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
  #get cost_history, accuracy history data:
  cost_history = np.append(cost_history,sess.run(cross_entropy,feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 0.5}))
  acc_history = np.append(acc_history,sess.run(accuracy,feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 0.5}))
          
#Measure Training Time:-------------------------------------------------       
t2_1 = datetime.datetime.now()
print("training time: " + str(t2_1-t1_1))

########################################################################
#testing different datasets:
########################################################################
  
##Testing: MNIST data

#(1) test all at once (needs big (4-6) GRAM memory):--------------------
#print("MNIST data: test accuracy %g"%accuracy.eval(feed_dict={
      #x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

#(2) test in batches:---------------------------------------------------
t1_2 = datetime.datetime.now()
test_size = 10000
correct = 0
batsize_test=5000 #use: 1000 for quadrok620// 5000 for i5,6gb ram

for step in range(int(test_size / batsize_test)):
  offset = step * batsize_test
  #print ('offset',offset)
  #print (offset + batsize_test)
  predictions = accuracy.eval(feed_dict={
    x: mnist.test.images[offset:(offset + batsize_test),:], 
    y_: mnist.test.labels[offset:(offset + batsize_test),:], keep_prob: 1.0})
  #print("MNIST data: batch_test accuracy %g"%predictions)
  correctbatch = np.append(correctbatch,predictions)
#print('MNIST data: test accuracy %.2f%%' % np.mean(correctbatch))
print("MNIST data: test accuracy %.3f"%np.mean(correctbatch, axis=0))  
     
#print np.info(mnist.test.images)    
#print np.info(mnist.test.labels)

t2_2 = datetime.datetime.now()
print("MNIST test time: " + str(t2_2-t1_2))

########################################################################

##Testing: Terminal data
#print("Terminal data: test accuracy %g"%accuracy.eval(feed_dict={
   # x: batch_terminal[0], y_: batch_terminal[1], keep_prob: 1.0}))

#test a subset of Terminal data:
#print("Terminal data: test accuracy term subset %g"%accuracy.eval(feed_dict={
#    x: termvec[0:1], y_: inputlabel[0:1], keep_prob: 1.0}))
  
#for bb in range(0, batsizeterminal ):
#    print 'Im value:', int(terminallabel[bb]),'Rec:', int(accuracy.eval(feed_dict={
#    x: termvec[bb+0:bb+1], y_: inputlabel[bb+0:bb+1], keep_prob: 1.0}))

#for bb in range(0, batsizeterminal ):
    #print (int(accuracy.eval(feed_dict={x: termvec[bb+0:bb+1], y_: inputlabel[bb+0:bb+1], keep_prob: 1.0})))


#show with Matplotlib:--------------------------------------------------
#bb=5
#plt.imshow(np.reshape(termvec[bb+0:bb+1],(28,28)), 
		#cmap = 'gray', interpolation = 'bicubic')
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#plt.show()
#print inputlabel[bb+0:bb+1]

#print np.info(mnist.test.images)    
#print np.info(mnist.test.labels)  

########################################################################

#plot settings:
#plot loss function-----------------------------------------------------
plt.figure(1, figsize=(8,8))
plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.subplot(211)
plt.plot(range(len(cost_history)),cost_history, color='b')
plt.axis([0,epochs,0,np.max(cost_history)])
plt.title('Cross Entropy Training Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
#plot accuracy function-------------------------------------------------
plt.subplot(212)
plt.plot(range(len(acc_history)),acc_history, color='g')
plt.axis([0,epochs,0,np.max(acc_history)])
plt.title('Training Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy percentage')

plt.show()
#plt.hold(False)


########################################################################
# opt. - export model weights to disk:
########################################################################
#define storage path for model:
#model_path = "/tmp/modelsaved/CNN5000e_1.ckpt"
#save model:
#save_path = saver.save(sess, model_path)
#print("Model saved in file: %s" % save_path)


