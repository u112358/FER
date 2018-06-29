import tensorflow as tf
import numpy as np

input_patches = tf.placeholder(dtype=tf.float32, shape=[68, 64, 64, 3])
input_gt = tf.placeholder(dtype=tf.int32, shape=[68, 2])

conv1 = tf.layers.conv2d(inputs=input_patches, kernel_size=(3, 3), filters=16, strides=1, padding='SAME')
pool1 = tf.layers.max_pooling2d(inputs=conv1, strides=2, pool_size=(2, 2))
conv2 = tf.layers.conv2d(inputs=pool1, kernel_size=(2, 2), filters=32, strides=1, padding='SAME')
pool2 = tf.layers.max_pooling2d(inputs=conv2, strides=2, pool_size=(2, 2))
conv3 = tf.layers.conv2d(inputs=pool2, kernel_size=(2, 2), filters=64, strides=1, padding='SAME')
pool3 = tf.layers.max_pooling2d(inputs=conv3, strides=2, pool_size=(2, 2))
flattened = tf.layers.flatten(pool3)
fc1 = tf.layers.dense(flattened, 10)
fc2 = tf.layers.dense(fc1, 2)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    fake_data = np.random.random((68, 64, 64, 3))
    fake_gt = np.random.randint(1, size=(68,2))
    print(sess.run(pool2, feed_dict={input_patches: fake_data, input_gt: fake_gt}))
    print('done')
    loss = tf.nn.l2_loss(fake_gt-fc2)
    print('done again!')
