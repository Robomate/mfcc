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

input_size = 1
batch_size = 1
time_steps = 2 #time steps
hidden_dim = 1
USE_LSTM = True

########################################################################
#~ print("===========================================")
#~ print("             numpy LSTM               ")
#~ print("===========================================")
#~ synapse_0 = 0.2*np.ones((input_size,hidden_dim), dtype=np.float32)
#~ synapse_h = 0.3*np.ones((hidden_dim,hidden_dim), dtype=np.float32)

#~ print("input U weight matrix")
#~ print(synapse_0)
#~ print(synapse_0.shape)
#~ print("input hidden W weight matrix")
#~ print(synapse_h)
#~ print(synapse_h.shape)

#~ print("concatenate U and W to weight_h matrix")
#~ weight_conc = np.concatenate((synapse_0,synapse_h), axis=0)
#~ print(weight_conc)
#~ print(weight_conc.shape)

vec_i = 1*np.ones((1,input_size ), dtype=np.float32) # define input
#~ vec_h = np.zeros((1,hidden_dim ), dtype=np.float32) #hidden state from tf
#~ print(vec_h)
#~ print(vec_h.shape)
#~ print("concatenate vector: hidden and input")
#~ vec_tot = np.concatenate((vec_i,vec_h), axis=1)
#~ print(vec_tot)
#~ print(vec_tot.shape)

#~ print("result")
#~ out_h = np.tanh(np.dot(vec_tot,weight_conc), dtype=np.float32)
#~ print(out_h)
#~ print(out_h.shape)

########################################################################
print("===========================================")
print("             tensorflow LSTM              ")
print("===========================================")


# Batch size x time steps x features.
data = tf.placeholder(tf.float32, [None, time_steps, input_size])
# define cell: RNN

if USE_LSTM:
    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, state_is_tuple=True)
else:
    cell = tf.nn.rnn_cell.BasicRNNCell(hidden_dim)
    
with tf.variable_scope("Rnn"):
	output, states = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

sess = tf.Session()
sess.run(tf.global_variables_initializer()) 

#feedtrain = {data: np.ones((batch_size, time_steps, input_size))}
feed = np.reshape(vec_i, [1,batch_size,input_size])
feed2 = np.concatenate((feed,feed), axis=1)
print("input:")
print(feed2)
print(feed2.shape)
feedtrain = {data: feed2}


print("pre init Weights:")
lstm_outx = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="Rnn")
lstm_out = sess.run(lstm_outx)
print(len(lstm_out))
print(lstm_outx[0])
print(lstm_out[0])
print(lstm_out[0].shape)
print(lstm_outx[1])
print(lstm_out[1])
print(lstm_out[1].shape)



#~ #assign new values:
#~ print("assign: Rnn weights")
#~ #get the tf.variable of the weights
#~ lstm_assign = lstm_outx[0].assign(weight_conc)
#~ #sess.run is needed to really assign values to matrices!!
#~ newval_assigned = sess.run(lstm_assign)
#~ #print (newval_assigned)

#~ lstm_outx2 = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="Rnn")
#~ lstm_out2 = sess.run(lstm_outx2)
#~ #print(len(lstm_out2))
#~ print("2 weight matrices")
#~ print(lstm_out2[0])
#~ print(lstm_out2[0].shape)
#~ print("bias")
#~ print(lstm_out2[1])
#~ print(lstm_out2[1].shape)
	
print("result: ouput")	
result2 = sess.run(output, feed_dict=feedtrain)	
print (result2)
print (result2[0].shape)	
    
print("result: states")	
result3 = sess.run(states, feed_dict=feedtrain)	
print (result3)
print (result3[0].shape)	
    

#~ lstm_out3 = sess.run(lstm_outx2)
#~ #print(len(lstm_out2))
#~ print("2 weight matrices")
#~ print(lstm_out3[0])
#~ print(lstm_out3[0].shape)


#~ print("result: ouput")
#~ feedtrain = {data: 23*np.ones([1,2,5])}
#~ result2 = sess.run(output, feed_dict=feedtrain)	
#~ print (result2)
#~ print (result2[0].shape)





