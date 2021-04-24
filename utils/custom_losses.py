import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import Loss
from tensorflow.keras.losses import binary_crossentropy


class DiceLoss(Loss):
    def __init__(self, batch_size, bce=False, smooth=1):
        """
        Dice loss is derived from: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch

        This also helped for some parts: https://towardsdatascience.com/creating-custom-loss-functions-using-tensorflow-2-96c123d5ce6c

        Current implementation is not using thresholding because only cast based thresholding is
        possible and it's not differentiable.

        :param batch_size: Batch size for iteration over batch dimension and calculation of mean value.
        :param smooth: Smoothing parameter
        """
        super().__init__()
        self.batch_size = batch_size
        self.smooth = smooth
        self.bce = bce

    def call(self, y_true, y_pred):
        dice = tf.constant(0.)
        bce_output = 0.

        for i in range(self.batch_size):
            y_true_f = tf.reshape(y_true[i, :, :, :], [-1])
            y_pred_f = tf.reshape(y_pred[i, :, :, :], [-1])

            intersection = tf.reduce_sum(y_true_f * y_pred_f)

            dice += (2. * intersection + self.smooth) / (
                    tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + self.smooth)

        if self.bce:
            bce_output = binary_crossentropy(y_true, y_pred, from_logits=False)

        return 1. - (dice / self.batch_size) + bce_output

    def get_config(self):
        """Returns the config dictionary for a `Loss` instance."""
        return {'reduction': self.reduction, 'name': self.name}
