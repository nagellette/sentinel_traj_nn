import tensorflow.keras.backend as K

"""
Dice loss is derived from: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
"""


def dice_loss(y_true, y_pred, smooth=1):
    """
    Dice loss function.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return 1 - dice
