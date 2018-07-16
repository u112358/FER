import tensorflow as tf
import numpy as np
from utils.image_processor import *


class FLDModel():
    def __init_(self):
        # patch images around prevision landmarks to pass into the Network to extract feature
        self.input_patches = tf.placeholder(dtype=tf.float32, shape=[68, 64, 64, 3], name='input_patches ')
        # ground truth of the landmarks, which are read from .pts files.
        self.input_gt = tf.placeholder(dtype=tf.float32, shape=[68, 2], name='input_gt')
        # current (prevision) landmarks.
        self.input_loc = tf.placeholder(dtype=tf.float32, shape=[68, 2], name='input_loc')

        self.init_loc = np.zeros(shape=[68, 2])

        self.epoch = 10000
        self.batch_size = 50
        self.lr = 1e-5

        self.features = self.extract_features(self.input_patches)
        self.delta_loc = self.predict_delta_loc(self.features)
        self.loss = tf.reduce_mean(tf.squared_difference(self.input_gt - (self.init_loc + self.delta_loc)))
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def extract_features(self, input_patches):
        """
        extract features of each patch
        """
        conv1 = tf.layers.conv2d(inputs=input_patches, kernel_size=(3, 3), filters=16, strides=1, padding='SAME')
        pool1 = tf.layers.max_pooling2d(inputs=conv1, strides=2, pool_size=(2, 2))
        conv2 = tf.layers.conv2d(inputs=pool1, kernel_size=(2, 2), filters=32, strides=1, padding='SAME')
        pool2 = tf.layers.max_pooling2d(inputs=conv2, strides=2, pool_size=(2, 2))
        conv3 = tf.layers.conv2d(inputs=pool2, kernel_size=(2, 2), filters=64, strides=1, padding='SAME')
        pool3 = tf.layers.max_pooling2d(inputs=conv3, strides=2, pool_size=(2, 2))
        flattened = tf.layers.flatten(pool3)

        return flattened

    def predict_delta_loc(self, features):
        """
        predict incremental (delta_loc) of the current location
        """
        fc1 = tf.layers.dense(features, 10)
        fc2 = tf.layers.dense(fc1, 2)

        return fc2


def main():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        IP = image_processor()
        model = FLDModel()
        model.init_loc = IP.get_init_loc()

        for epoch in range(model.epoch):
            image, gt = IP.get_next_image_and_gt()
            loc = model.init_loc

            for i in range(100):  # iterations
                patches = IP.get_patches(image, loc)
                _, loss, delta_loc = sess.run([model.train_op, model.loss, model.delta_loc],
                                              feed_dict={model.input_patches: patches, model.input_gt: gt,
                                                         model.init_loc: loc})
                loc = loc + delta_loc
                print('Epoch:%d\t Step:%d out of %d, loss=%lf\n' % epoch, i, 100, loss)
                if loss < 1e-3:
                    print('loss satisfies the criterion, jump to the next image! \n')
                    break


    if __name__ == "__main__":
        main()
