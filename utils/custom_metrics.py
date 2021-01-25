from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.metrics import Metric

MeanIoU_THRESHOLD = 0.5


@tf.function
def mean_iou(y_true, y_pred):
    y_pred = y_pred[:, :, :, 0]
    y_true = y_true[:, :, :, 0]
    y_pred = K.cast(K.greater(y_pred, 0.5), dtype='float32')  # .5 is the threshold
    inter = K.sum(K.sum(y_true * y_pred))
    union = K.sum(K.sum(y_true + y_pred))

    return K.mean((inter + K.epsilon()) / (union + K.epsilon()))


class MeanIoU(Metric):

    def __init__(self, num_classes=2, threshold=0.5, name=None, dtype=None):
        super(MeanIoU, self).__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        self.threshold = threshold

        self.mean_iou = self.add_weight(name="mean_iou", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):

        if y_true.shape.ndims > 1:
            y_true = tf.reshape(y_true, [-1])

        if y_pred.shape.ndims > 1:
            y_pred = tf.reshape(y_pred, [-1])

        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(tf.math.greater(y_pred, self.threshold), dtype=tf.float32)

        intersection = tf.math.reduce_sum(y_true * y_pred)
        union = tf.math.reduce_sum(y_true + y_pred)

        return self.mean_iou.assign(tf.reduce_mean((intersection + K.epsilon()) / (union + K.epsilon())))

    def result(self):
        return self.mean_iou
