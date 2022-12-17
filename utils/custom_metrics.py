from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.metrics import Metric

import numpy as np


def create_boundary(array):
    # get input shape
    buffer_row, buffer_col = array.shape[0], array.shape[1]

    # create output canvas
    new_array = np.ones((buffer_row, buffer_col))

    # loop over pixels, update to boundary pixels
    for i in range(0, buffer_row):
        for j in range(0, buffer_col):
            if array[i][j] == 0:
                new_array[i][j] = 1
                if i - 1 >= 0 and array[i - 1][j] != 0:
                    new_array[i - 1][j] = 0
                if j - 1 >= 0 and array[i][j - 1] != 0:
                    new_array[i][j - 1] = 0
                if i - 1 >= 0 and j - 1 >= 0 and array[i - 1][j - 1] != 0:
                    new_array[i - 1][j - 1] = 0
                if i + 1 < buffer_row and array[i + 1][j] != 0:
                    new_array[i + 1][j] = 0
                if j + 1 < buffer_col and array[i][j + 1] != 0:
                    new_array[i][j + 1] = 0
                if i + 1 < buffer_row and j + 1 < buffer_col and array[i + 1][j + 1] != 0:
                    new_array[i + 1][j + 1] = 0
                if i - 1 >= 0 and j + 1 < buffer_col and array[i - 1][j + 1] != 0:
                    new_array[i - 1][j + 1] = 0
                if i + 1 < buffer_row and j - 1 >= 0 and array[i + 1][j - 1] != 0:
                    new_array[i + 1][j - 1] = 0

    return new_array.astype('uint8')


class MeanIoU(Metric):

    def __init__(self, batch_size, num_classes=2, threshold=0.5, name=None, dtype=None):
        """
        Mean intersection over union metric.

        :param batch_size: size of batch for iteration to calculate mean value
        :param num_classes: number of classes
        :param threshold: threshold for defining class from the achieved probability
        :param name:
        :param dtype:
        """
        super(MeanIoU, self).__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        self.threshold = threshold
        self.batch_size = batch_size

        self.mean_iou = self.add_weight(name="mean_iou", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):

        iou_sum = tf.constant(0.)

        for i in range(self.batch_size):

            y_true_f = y_true[i, :, :, :]
            y_pred_f = y_pred[i, :, :, :]

            if y_true_f.shape.ndims > 1:
                y_true_f = tf.reshape(y_true_f, [-1])

            if y_pred_f.shape.ndims > 1:
                y_pred_f = tf.reshape(y_pred_f, [-1])

            y_true_f = tf.cast(y_true_f, dtype=tf.float32)
            y_pred_f = tf.cast(tf.math.greater(y_pred_f, self.threshold), dtype=tf.float32)

            intersection = tf.math.reduce_sum(y_true_f * y_pred_f)
            union = tf.math.reduce_sum(y_true_f + y_pred_f)

            iou_sum = iou_sum + ((intersection + K.epsilon()) / (union + K.epsilon()))

        return self.mean_iou.assign(iou_sum / self.batch_size)

    def result(self):
        return self.mean_iou
