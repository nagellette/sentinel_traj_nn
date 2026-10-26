import tensorflow as tf


class ConstructOptimizer:
    def __init__(self, optimizer_name, l_rate, decay, momentum, nesterov):
        self.optimizer_name = optimizer_name
        self.l_rate = l_rate
        self.decay = decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.optimizer = self.set_optimizer()

    def set_optimizer(self):
        if self.optimizer_name == "SGD":
            print("Setting optimizer as SGD with "
                  "l_rate: {}, decay: {}, momentum: {}, nesterov: {}.".format(self.l_rate, self.decay, self.momentum,
                                                                              self.nesterov))
            # legacy optimizers keep TF 2.5 behaviour; the new SGD rejects `decay`
            return tf.keras.optimizers.legacy.SGD(learning_rate=self.l_rate,
                                                  decay=self.decay,
                                                  momentum=self.momentum,
                                                  nesterov=self.nesterov)

        elif self.optimizer_name == "adam":
            print("Setting optimizer as Adam with l_rate: {}.".format(self.l_rate))
            return tf.keras.optimizers.legacy.Adam(learning_rate=self.l_rate)

        else:
            print("Optimizer is not defined, please define in utils.constract_lost_function.py")

    def get_optimizer(self):
        return self.optimizer
