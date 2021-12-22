import numpy as np
import tensorflow as tf

ghost = tf.placeholder(tf.float32, shape=[3, 2])
temp1 = [[1, 2], [4, 2], [4, 5]]

sess = tf.Session()
print(sess.run([ghost], feed_dict={ghost: temp1}))