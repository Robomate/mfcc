#https://stackoverflow.com/questions/40325012/get-value-of-weights-and-bias-from-rnn-cell
#https://stackoverflow.com/questions/40318812/tensorflow-rnn-weight-matrices-initialization
#https://github.com/tensorflow/tensorflow/issues/3115
#https://stackoverflow.com/questions/40318812/tensorflow-rnn-weight-matrices-initialization
#https://stackoverflow.com/questions/42962281/can-you-manually-set-tensorflow-lstm-weights
import tensorflow as tf
import numpy as np
import os
# remove warnings from tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

input_size = 5
batch_size = 2
max_length = 1

cell = tf.nn.rnn_cell.BasicRNNCell(num_units = 4)

# Batch size x time steps x features.
data = tf.placeholder(tf.float32, [None, max_length, input_size])
with tf.variable_scope("Rnn"):
	output, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

sess = tf.Session()
sess.run(tf.global_variables_initializer()) 

feedtrain = {data: np.ones((batch_size, max_length, input_size))}
result = sess.run([output], feed_dict=feedtrain)
print (result)
print (result[0].shape)

#print values
variables_names =[v.name for v in tf.trainable_variables()]
print(variables_names)
values = sess.run(variables_names)
for k,v in zip(variables_names, values):
	print(k)
	print(v)
	
print("==================================")
print("Rnn")
lstm_outx = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="Rnn")
lstm_out = sess.run(lstm_outx)
print(len(lstm_out))
print(lstm_outx[0])
print(lstm_out[0])
print(lstm_outx[1])
print(lstm_out[1])
print(lstm_out[0].shape)
print(lstm_out[1].shape)
#~ print(lstm_out[2].shape)
#~ print(lstm_out[3].shape)
#~ print(lstm_out[4].shape)
#~ print(lstm_out[5].shape)

#assign new values:
print("assign: Rnn")
#get the tf.variable of the weights
lstm_assign = lstm_outx[0].assign(np.ones(lstm_out[0].shape) * 2)
#sess.run is needed to really assign values to matrices!!
print (sess.run(lstm_assign))

lstm_outx2 = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="Rnn")
lstm_out2 = sess.run(lstm_outx2)
print(len(lstm_out2))
print(lstm_out2[0])
print(lstm_out2[1])
	
	
result2 = sess.run([output], feed_dict=feedtrain)	
print (result2)
print (result2[0].shape)	
    
	



	
