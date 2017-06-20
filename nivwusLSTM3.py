#link:https://gist.github.com/nivwusquorum/b18ce332bde37e156034e5d3f60f8a23
#https://github.com/pkmital/tensorflow_tutorials
#https://github.com/BinRoot/TensorFlow-Book/blob/master/ch06_hmm/hmm.py

"""Short and sweet LSTM implementation in Tensorflow.
Motivation:
When Tensorflow was released, adding RNNs was a bit of a hack - it required
building separate graphs for every number of timesteps and was a bit obscure
to use. Since then TF devs added things like `dynamic_rnn`, `scan` and `map_fn`.
Currently the APIs are decent, but all the tutorials that I am aware of are not
making the best use of the new APIs.
Advantages of this implementation:
- No need to specify number of timesteps ahead of time. Number of timesteps is
  infered from shape of input tensor. Can use the same graph for multiple
  different numbers of timesteps.
- No need to specify batch size ahead of time. Batch size is infered from shape
  of input tensor. Can use the same graph for multiple different batch sizes.
- Easy to swap out different recurrent gadgets (RNN, LSTM, GRU, your new
  creative idea)
"""


import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


map_fn = tf.map_fn

################################################################################
##                           DATASET GENERATION                               ##
##                                                                            ##
##  The problem we are trying to solve is adding two binary numbers. The      ##
##  numbers are reversed, so that the state of RNN can add the numbers        ##
##  perfectly provided it can learn to store carry in the state. Timestep t   ##
##  corresponds to bit len(number) - t.                                       ##
################################################################################

def as_bytes(num, final_size):
    res = []
    for _ in range(final_size):
        res.append(num % 2)
        num //= 2
    return res

def generate_example(num_bits):
    a = random.randint(0, 2**(num_bits - 1) - 1)
    b = random.randint(0, 2**(num_bits - 1) - 1)
    res = a + b
    return (as_bytes(a,  num_bits),
            as_bytes(b,  num_bits),
            as_bytes(res,num_bits))

def generate_batch(num_bits, batch_size):
    """Generates instance of a problem.
    Returns
    -------
    x: np.array
        two numbers to be added represented by bits.
        shape: b, i, n
        where:
            b is bit index from the end
            i is example idx in batch
            n is one of [0,1] depending for first and
                second summand respectively
    y: np.array
        the result of the addition
        shape: b, i, n
        where:
            b is bit index from the end
            i is example idx in batch
            n is always 0
    """
    x = np.empty((num_bits, batch_size, 2))
    y = np.empty((num_bits, batch_size, 1))

    for i in range(batch_size):
        a, b, r = generate_example(num_bits)
        x[:, i, 0] = a
        x[:, i, 1] = b
        y[:, i, 0] = r
    #print(x.shape)
    return x, y


################################################################################
##                           GRAPH DEFINITION                                 ##
################################################################################

INPUT_SIZE    = 2       # 2 bits per timestep
RNN_HIDDEN    = 20
OUTPUT_SIZE   = 1       # 1 bit per timestep
TINY          = 1e-6    # to avoid NaNs in logs
LEARNING_RATE = 0.01

USE_LSTM = True

inputs  = tf.placeholder(tf.float32, (None, None, INPUT_SIZE))  # (time, batch, in)
outputs = tf.placeholder(tf.float32, (None, None, OUTPUT_SIZE)) # (time, batch, out)



## Here cell can be any function you want, provided it has two attributes:
#     - cell.zero_state(batch_size, dtype)- tensor which is an initial value
#                                           for state in __call__
#     - cell.__call__(input, state) - function that given input and previous
#                                     state returns tuple (output, state) where
#                                     state is the state passed to the next
#                                     timestep and output is the tensor used
#                                     for infering the output at timestep. For
#                                     example for LSTM, output is just hidden,
#                                     but state is memory + hidden
# Example LSTM cell with learnable zero_state can be found here:
#    https://gist.github.com/nivwusquorum/160d5cf7e1e82c21fad3ebf04f039317
if USE_LSTM:
    cell = tf.nn.rnn_cell.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)
else:
    cell = tf.nn.rnn_cell.BasicRNNCell(RNN_HIDDEN)

# Create initial state. Here it is just a constant tensor filled with zeros,
# but in principle it could be a learnable parameter. This is a bit tricky
# to do for LSTM's tuple state, but can be achieved by creating two vector
# Variables, which are then tiled along batch dimension and grouped into tuple.
batch_size    = tf.shape(inputs)[1]
initial_state = cell.zero_state(batch_size, tf.float32)

# Given inputs (time, batch, input_size) outputs a tuple
#  - outputs: (time, batch, output_size)  [do not mistake with OUTPUT_SIZE]
#  - states:  (time, batch, hidden_size)
with tf.variable_scope("LSTM"):
	rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, time_major=True)

# project output from rnn output size to OUTPUT_SIZE. Sometimes it is worth adding
# an extra layer here.
final_projection = lambda x: layers.linear(x, num_outputs=OUTPUT_SIZE, activation_fn=tf.nn.sigmoid)

# apply projection to every timestep.
predicted_outputs = map_fn(final_projection, rnn_outputs)


#same as bnorm: outputs_collections='vars'
#same beta and gamma: variables_collections=['vars2'] requires list of strings
#updates_collections=tf.GraphKeys.UPDATE_OPS
with tf.variable_scope("Bnorm"):
	bnorm = tf.contrib.layers.batch_norm(predicted_outputs,center=True, scale=True,
	is_training=True, updates_collections=tf.GraphKeys.UPDATE_OPS,variables_collections=['vars2'], 
	outputs_collections='vars', decay=0.999, zero_debias_moving_mean=False)
	
with tf.variable_scope("Lnorm"):
	lnorm = tf.contrib.layers.layer_norm(predicted_outputs,center=True, scale=True,
	trainable=True,variables_collections=['varsx'], outputs_collections=None)

# compute elementwise cross entropy.
error = -(outputs * tf.log(predicted_outputs + TINY) + (1.0 - outputs) * tf.log(1.0 - predicted_outputs + TINY))
error = tf.reduce_mean(error)

# optimize
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
#~ with tf.control_dependencies(update_ops):
train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error)

#train_fn = tf.contrib.keras.optimizers.Adamax(lr=LEARNING_RATE).get_gradients(error)


# assuming that absolute difference between output and correct answer is 0.5
# or less we can round it to the correct output.
accuracy = tf.reduce_mean(tf.cast(tf.abs(outputs - predicted_outputs) < 0.5, tf.float32))
#tf.add_to_collection('vars', accuracy)

################################################################################
##                           TRAINING LOOP                                    ##
################################################################################

NUM_BITS = 10
ITERATIONS_PER_EPOCH = 100
BATCH_SIZE = 16

valid_x, valid_y = generate_batch(num_bits=NUM_BITS, batch_size=100)

session = tf.Session()
# save and restore all the variables:
saver = tf.train.Saver()
# For some reason it is our job to do this:
session.run(tf.global_variables_initializer()) 

for epoch in range(1000):
	epoch_error = 0
	for _ in range(ITERATIONS_PER_EPOCH):
		# here train_fn is what triggers backprop. error and accuracy on their
		# own do not trigger the backprop.
		x, y = generate_batch(num_bits=NUM_BITS, batch_size=BATCH_SIZE)
		epoch_error += session.run([error, train_fn], {
			inputs: x,
			outputs: y,
		})[0]
	epoch_error /= ITERATIONS_PER_EPOCH
	feedvalid = {inputs:  valid_x, outputs: valid_y }
	valid_accuracy = session.run(accuracy, feedvalid)
	####################################################################
	#get variables lstm:
	####################################################################
	print("output")
	outputs = session.run(rnn_outputs,feed_dict=feedvalid)
	print(outputs)
	print(outputs.shape)
	print("states") #for Lstm tuple
	statess = session.run(rnn_states,feed_dict=feedvalid)
	print(statess)
	print(len(statess))
	print(statess[0])
	print(statess[1])
	print(statess[0].shape)
	print(statess[1].shape)
	
	print("lstm")
	lstm_outx = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="LSTM")
	lstm_out = session.run(lstm_outx,feed_dict=feedvalid)
	print(lstm_outx)
	print(len(lstm_out))
	print(lstm_out[0].shape)
	print(lstm_out[1].shape)
	print(lstm_out[2].shape)
	print(lstm_out[3].shape)
	print(lstm_out[4].shape)
	print(lstm_out[5].shape)
	
	
	print("===========")
	####################################################################
	#get variables bnorm:
	####################################################################
	predicted_outputs
	predicted_outputs0 = session.run(predicted_outputs,feed_dict=feedvalid)
	print(len(session.run(bnorm,feed_dict=feedvalid)))
	print("predicted_outputs")
	print(len(predicted_outputs0))
	print(predicted_outputs0[0].shape)
	print(predicted_outputs0[9].shape)
	
	print("update_ops")
	print("mov_mean, mov_var")
	update_ops0 = session.run(update_ops,feed_dict=feedvalid)
	print(len(update_ops0))
	print(update_ops0[0])
	print(update_ops0[1])
	
	
	#~ print("update_collections")
	#~ print("beta, gamma")
	#~ all_vars8 = tf.get_collection('vars3')
	#~ all_vars9 = session.run(all_vars8,feed_dict=feedvalid)
	#~ print(len(all_vars9))
	#~ print(all_vars9[0])
	#~ print(all_vars9[1])
	
	
	print("variables_collections")
	print("mov_mean, mov_var,beta, gamma")
	all_vars = tf.get_collection('vars2')
	all_vars4 = session.run(all_vars,feed_dict=feedvalid)
	print(all_vars)
	print(len(all_vars4))
	print(all_vars4[0])
	print(all_vars4[1])
	print(all_vars4[2])
	print(all_vars4[3])
	
	print("tf.GraphKeys.TRAINABLE_VARIABLES")
	trainable_var_key = tf.GraphKeys.TRAINABLE_VARIABLES
	all_vars2 = tf.get_collection(key=trainable_var_key, scope="Bnorm")
	print("===========")
	print("beta, gamma")
	print(all_vars2)
	all_vars3 = session.run(all_vars2,feed_dict=feedvalid)
	print(all_vars3[0]) #betas = shift
	print(all_vars3[1]) #gammas = scale
	print("===========")
	
	#get other variables
	print("tf.GraphKeys.GLOBAL_VARIABLES")
	print("mov_mean, mov_var,beta, gamma")
	all_vars10 = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="Bnorm")
	all_vars11 = session.run(all_vars10,feed_dict=feedvalid)
	print(all_vars10)
	print(len(all_vars11))
	print(all_vars11)
	####################################################################
	#get variables lnorm:
	####################################################################
	print("===========")
	print("lnorm.variables_collections")
	print("beta, gamma")
	all_vars44 = tf.get_collection('varsx')
	all_vars45 = session.run(all_vars44,feed_dict=feedvalid)
	print(all_vars44)
	print(len(all_vars45))
	print(all_vars45[0])
	print(all_vars45[1])
	####################################################################
	#assign variables to  lnorm:
	####################################################################
	#get the tf.variable
	beta_ass = all_vars44[0].assign(np.ones(all_vars44[0].shape) * 7)
	#sess.run is needed to really assign values to matrices!!
	print("assigned: beta")
	print (session.run(beta_ass))
	
	print("Lnorm: tf.GraphKeys.TRAINABLE_VARIABLES")
	trainables_var_key = tf.GraphKeys.TRAINABLE_VARIABLES
	all_vars23 = tf.get_collection(key=trainables_var_key, scope="Lnorm")
	print("beta, gamma")
	print(all_vars23)
	all_vars33 = session.run(all_vars23,feed_dict=feedvalid)
	print(all_vars33[0]) #betas = shift
	print(all_vars33[1]) #gammas = scale
	#get other variables
	print("Lnorm: tf.GraphKeys.GLOBAL_VARIABLES")
	print("beta, gamma")
	all_vars15 = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="Lnorm")
	all_vars16 = session.run(all_vars15,feed_dict=feedvalid)
	print(all_vars15)
	print(len(all_vars16))
	print(all_vars16)
	print("===========")
	print("===========")
	
	#print ("Epoch %d, train error: %.2f, valid accuracy: %.1f %%" % (epoch, epoch_error, valid_accuracy * 100.0))
	
