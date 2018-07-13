import tensorflow as tf

X = tf.constant([1.3, 2.5, -3.1, 2.2, -0.6, -6.2])
Y = tf.constant([1, 2, 3, 4, 5, 6])
dataset = tf.data.Dataset.from_tensor_slices(X)
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
