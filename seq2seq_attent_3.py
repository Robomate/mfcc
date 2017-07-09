
#old
#https://www.tensorflow.org/api_docs/python/tf/contrib/legacy_seq2seq/basic_rnn_seq2seq

#new
#https://www.tensorflow.org/api_guides/python/contrib.seq2seq
#https://stackoverflow.com/questions/43656938/attention-mechanism-for-sequence-classification-seq2seq-tensorflow-r1-1

import tensorflow as tf
import numpy as np
import os


encoder_inputs = []
decoder_inputs = []
n_hidden = 512
for i in range(350):  
    encoder_inputs.append(tf.placeholder(tf.float32, shape=[None,512],
                                          name="encoder{0}".format(i)))

for i in range(45):
    decoder_inputs.append(tf.placeholder(tf.float32, shape=[None,512],
                                         name="decoder{0}".format(i)))

########################################################################
# classical seq2seq model
########################################################################
cell = tf.nn.rnn_cell.LSTMCell(n_hidden,state_is_tuple=True)
#Basic RNN sequence-to-sequence model                                
outputs, states = tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(encoder_inputs, decoder_inputs, cell)

########################################################################
# attention seq2seq model
########################################################################

############################# Encoder ##################################
encoder_cell = rnn.BasicLSTMCell(n_hidden)
encoder_outputs, encoder_states = tf.nn.dynamic_rnn(encoder_cell, 
                                  x_input,dtype=tf.float32)

############################# Decoder ##################################
decoder_cell = rnn.BasicLSTMCell(n_hidden)
# Attention mechanism
attention_mechanism = tf.contrib.seq2seq.LuongAttention(n_hidden, encoder_outputs)
#memory [batch_size, memory_max_time, memory_depth]
#attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(n_hidden, memory)
attn_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, 
            attention_mechanism, attention_size=n_hidden)

# Initial attention
attn_zero = attn_cell.zero_state(batch_size=tf.shape(x)[0], dtype=tf.float32)
init_state = attn_zero.clone(cell_state=states[0])

# Helper function
helper = tf.contrib.seq2seq.TrainingHelper(inputs = ???)

# Decoding
decoder = tf.contrib.seq2seq.BasicDecoder(cell=attn_cell,
             helper=helper,
             initial_state=init_state)

decoder_outputs, decoder_states = tf.contrib.seq2seq.dynamic_decode(my_decoder)



#below is for training / inference decoder
#https://github.com/udacity/deep-learning/blob/master/seq2seq/sequence_to_sequence_implementation.ipynb
#https://stackoverflow.com/questions/43622778/tensorflow-v1-1-seq2seq?rq=1


# set up training decoder
with tf.variable_scope("decode"):
	# Helper for the training process. Used by BasicDecoder to read inputs.
	training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
														sequence_length=target_sequence_length,
														time_major=False)
	# Basic decoder
	training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
													   training_helper,
													   enc_state,
													   output_layer)
	# Perform dynamic decoding using the decoder
	training_decoder_output, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
																   impute_finished=True,
																   maximum_iterations=max_target_sequence_length)
  
# set up inference decoder
# reuses the same parameters trained by the training process
with tf.variable_scope("decode", reuse=True):
	start_tokens = tf.tile(tf.constant([target_letter_to_int['<GO>']], dtype=tf.int32), [batch_size], name='start_tokens')
	# Helper for the inference process.
	inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings,
															start_tokens,
															target_letter_to_int['<EOS>'])
	# Basic decoder
	inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
													inference_helper,
													enc_state,
													output_layer)
	# Perform dynamic decoding using the decoder
	inference_decoder_output, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
														impute_finished=True,
														maximum_iterations=max_target_sequence_length)
  
  
  
  
  
  
  
