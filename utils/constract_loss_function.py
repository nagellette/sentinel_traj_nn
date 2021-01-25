import tensorflow as tf
import utils.custom_losses as custom_losses


class ConstructLossFunction:
    def __init__(self, loss_function_name):
        self.loss_function_name = loss_function_name
        self.loss_function = self.set_loss_function()

    def set_loss_function(self):

        if self.loss_function_name == "dice":
            print("Setting loss function as Dice loss with output probabilities.")
            return custom_losses.dice_loss_soft

        elif self.loss_function_name == "dice_th":
            print("Setting loss function as Dice loss with threshold 0.5.")
            return custom_losses.dice_loss

        elif self.loss_function_name == "binary_crossentropy":
            print("Setting loss function as binary cross entropy loss.")
            return tf.keras.losses.BinaryCrossentropy(from_logits=True)

        else:
            print("Loss function is not defined, please define in utils.constract_lost_function.py")

    def get_loss_function(self):
        return self.loss_function
