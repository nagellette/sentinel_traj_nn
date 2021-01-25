import tensorflow as tf
import tensorflow.keras.backend as K

"""
Dice loss is derived from: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
"""


def dice_loss_soft(y_true, y_pred, smooth=1):
    """
    Dice loss function. (SOFT, without threshold)
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return 1 - dice


def dice_loss(y_true, y_pred, threshold=0.5, smooth=1):
    """
    Dice loss function.
    """
    print(y_pred)
    print(y_true)

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    print(y_pred_f)
    print(y_true_f)

    # y_true_f = tf.cast(y_true_f, dtype=tf.float32)
    y_pred_f = tf.cast(tf.math.greater(y_pred_f, threshold), dtype=tf.float32)

    print(y_pred_f)
    print(y_true_f)

    intersection = tf.math.reduce_sum(y_pred_f * y_true_f)

    dice = (2. * intersection + smooth) / (tf.math.reduce_sum(y_true_f) + tf.math.reduce_sum(y_pred_f) + smooth)

    return 1. - dice
