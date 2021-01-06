from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.metrics import Metric
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops

MeanIoU_THRESHOLD = 0.5

@tf.function
def mean_iou(y_true, y_pred):
    y_pred = y_pred[:, :, :, 0]
    y_true = y_true[:, :, :, 0]
    y_pred = K.cast(K.greater(y_pred, 0.5), dtype='float32')  # .5 is the threshold
    inter = K.sum(K.sum(y_true * y_pred))
    union = K.sum(K.sum(y_true + y_pred))

    return K.mean((inter + K.epsilon()) / (union + K.epsilon()))
