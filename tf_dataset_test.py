import tensorflow as tf
import numpy as np


T = tf.constant(np.zeros([20, 10, 24, 30, 15]))
print(T.get_shape())
T = tf.split(T, num_or_size_splits=3, axis=3)
print(T[0].get_shape())

X = tf.constant([1.3, 2.5, -3.1, 2.2, -0.6, -6.2])
Y = tf.constant([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0]])
dataset = tf.data.Dataset.from_tensor_slices((X, Y))
dataset = dataset.shuffle(buffer_size=10).batch(2).repeat(10)


iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)
sess.run(iterator.initializer)

# print(sess.run(next_element[0]))
print(sess.run(next_element))
print(sess.run(next_element))
print(sess.run(next_element))
print(sess.run(next_element))
print(sess.run(next_element))
print(sess.run(next_element))
print(sess.run(next_element))
