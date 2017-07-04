

import numpy as np
import tensorflow as tf
import os

# remove warnings from tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# init tf session :
sess = tf.InteractiveSession()
# start session:
sess.run(tf.global_variables_initializer()) 

# Graph definition
x = tf.sparse_placeholder(tf.float32)
y = tf.sparse_reduce_sum(x)

# Values to feed the sparse placeholder
indices = np.array([[0, 0], [0, 1], [0, 2]], dtype=np.int64)
values = np.array([1.0, 2.0, 32.0], dtype=np.float32) #labels (in sequence concatenated)
dense_shape = np.array([2,], dtype=np.int64)

# Option 2
print(sess.run(x, feed_dict={x: (indices, values, dense_shape)}))
#print(sess.run(y, feed_dict={x: (indices, values, dense_shape)}))


