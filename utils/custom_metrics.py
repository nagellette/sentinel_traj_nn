import numpy as np

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.metrics import Metric


class MeanIoUCustom(Metric):
    """
    Customized version of Mean Intersection over Union TODO: Not completed yet
    """

    def __init__(self, num_classes, threshold=0.5, name=None, dtype=None):
        super(MeanIoUCustom, self).__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        self.threshold = threshold

        # Variable to accumulate the predictions in the confusion matrix. Setting
        # the type to be `float64` as required by confusion_matrix_ops.
        self.total_cm = self.add_weight(
            'total_confusion_matrix',
            shape=(num_classes, num_classes),
            initializer=init_ops.zeros_initializer,
            dtype=dtypes.float64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the confusion matrix statistics.

        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.

        Returns:
          Update op.
        """

        y_true = math_ops.cast(y_true, self._dtype)
        y_pred = tf.where(y_pred >= self.threshold, tf.zeros_like(y_pred), tf.ones_like(y_pred))
        y_pred = math_ops.cast(y_pred, self._dtype)

        # Flatten the input if its rank > 1.
        if y_pred.shape.ndims > 1:
            y_pred = array_ops.reshape(y_pred, [-1])

        if y_true.shape.ndims > 1:
            y_true = array_ops.reshape(y_true, [-1])

        if sample_weight is not None:
            sample_weight = math_ops.cast(sample_weight, self._dtype)
            if sample_weight.shape.ndims > 1:
                sample_weight = array_ops.reshape(sample_weight, [-1])

        # Accumulate the prediction to current confusion matrix.
        current_cm = confusion_matrix.confusion_matrix(
            y_true,
            y_pred,
            self.num_classes,
            weights=sample_weight,
            dtype=dtypes.float64)
        return self.total_cm.assign_add(current_cm)

    def result(self):
        """Compute the mean intersection-over-union via the confusion matrix."""
        sum_over_row = math_ops.cast(
            math_ops.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
        sum_over_col = math_ops.cast(
            math_ops.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
        true_positives = math_ops.cast(
            array_ops.diag_part(self.total_cm), dtype=self._dtype)

        # sum_over_row + sum_over_col =
        #     2 * true_positives + false_positives + false_negatives.
        denominator = sum_over_row + sum_over_col - true_positives

        # The mean is only computed over classes that appear in the
        # label or prediction tensor. If the denominator is 0, we need to
        # ignore the class.
        num_valid_entries = math_ops.reduce_sum(
            math_ops.cast(math_ops.not_equal(denominator, 0), dtype=self._dtype))

        iou = math_ops.div_no_nan(true_positives, denominator)

        return math_ops.div_no_nan(
            math_ops.reduce_sum(iou, name='mean_iou'), num_valid_entries)

    def reset_states(self):
        K.set_value(self.total_cm, np.zeros((self.num_classes, self.num_classes)))

    def get_config(self):
        config = {'num_classes': self.num_classes}
        base_config = super(MeanIoUCustom, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
