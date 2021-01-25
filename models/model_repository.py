import sys

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, \
    Activation, Add
from tensorflow_core.python.keras.layers import AveragePooling2D, GlobalAveragePooling2D

from utils.constract_loss_function import ConstructLossFunction
from utils.construct_optimizer import ConstructOptimizer
from utils.get_metrics import get_metrics


class ModelRepository:
    def __init__(self,
                 model_name,
                 dim,
                 input_channels,
                 batch_size,
                 srcnn_count=0,
                 optimizer="SGD",
                 l_rate=0.001,
                 decay=1e-6,
                 momentum=0.9,
                 nesterov=True,
                 loss="dice"):
        """
        Collection of deep learning models for image segmentation.
        :param model_name:name of the model that'll run.
        :param dim: 2D dimension of the input image
        :param input_channels: channel count of the input image
        :param batch_size: size of a batch
        :param srcnn_count: number of srcnn layers - first srcnn_count images are applied, later not applied or ignored.
        :param optimizer: Optimizer
        :param l_rate: Learning rate.
        :param decay: decay parameter for SGD optimizer
        :param momentum: momentum parameter for SGD optimizer
        :param nesterov: nesterov parameter for SGD optimizer
        :param loss: loss function parameter
        """
        self.model_name = model_name
        self.dim = dim
        self.input_channels = input_channels
        self.batch_size = batch_size
        self.srcnn_count = srcnn_count
        self.optimizer = optimizer
        self.l_rate = l_rate
        self.decay = decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.loss = loss
        self.model = None

        self.loss_function = ConstructLossFunction(loss_function_name=self.loss).get_loss_function()
        self.optimizer = ConstructOptimizer(optimizer_name=self.optimizer, l_rate=self.l_rate, decay=self.decay,
                                            momentum=self.momentum, nesterov=self.nesterov).get_optimizer()

        # model to run
        if self.model_name == "test_model":
            self.test_model()
        elif self.model_name == "unet":
            self.unet(self.dim, self.input_channels, self.batch_size)
        elif self.model_name == "unetlight":
            self.unet_light(self.dim, self.input_channels, self.batch_size)
        elif self.model_name == "srcnn_unet":
            self.srcnn_unet(self.dim, self.input_channels, self.batch_size, self.srcnn_count)
        elif self.model_name == "resunet":
            self.resunet(self.dim, self.input_channels, self.batch_size)
        elif self.model_name == "resunetlight":
            self.resunet_light(self.dim, self.input_channels, self.batch_size)
        elif self.model_name == "dlinknet":
            self.dlinknet(self.dim, self.input_channels, self.batch_size)
        else:
            print(self.model_name + " not defined yet.")
            sys.exit()

    def get_model(self):
        return self.model

    def test_model(self):
        self.model = tf.keras.Sequential()
        # model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(100, activation='relu'))
        self.model.add(tf.keras.layers.Dense(100, activation='relu'))
        self.model.add(tf.keras.layers.Dense(100, activation='relu'))
        self.model.add(tf.keras.layers.Dense(10000, activation='softmax'))
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_function,
                           metrics=['accuracy'])

    def unet(self, dim, input_channels, batch_size):

        """
        Unet implementation:
        - https://arxiv.org/abs/1505.04597

        :param dim: dimension of inputs
        :param input_channels: number of bands/layers of input
        :param batch_size: # batches in the input
        :return: compiled model
        """

        inputs_layer = tf.keras.layers.Input((dim[0], dim[1], input_channels), batch_size=batch_size)
        conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(inputs_layer)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D((2, 2))(conv1)

        conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D((2, 2))(conv2)

        conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D((2, 2))(conv3)

        conv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv4)
        conv4 = BatchNormalization()(conv4)
        pool4 = MaxPooling2D((2, 2))(conv4)

        conv_middle = Conv2D(1024, (3, 3), activation="relu", padding="same")(pool4)
        conv_middle = BatchNormalization()(conv_middle)
        conv_middle = Conv2D(1024, (3, 3), activation="relu", padding="same")(conv_middle)
        conv_middle = BatchNormalization()(conv_middle)

        conv_t4 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(conv_middle)
        conc4 = concatenate([conv_t4, conv4])
        up_conv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(conc4)
        up_conv4 = BatchNormalization()(up_conv4)
        up_conv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(up_conv4)
        up_conv4 = BatchNormalization()(up_conv4)

        conv_t3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(up_conv4)
        conc3 = concatenate([conv_t3, conv3])
        up_conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(conc3)
        up_conv3 = BatchNormalization()(up_conv3)
        up_conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(up_conv3)
        up_conv3 = BatchNormalization()(up_conv3)

        conv_t2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(up_conv3)
        conc2 = concatenate([conv_t2, conv2])
        up_conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(conc2)
        up_conv2 = BatchNormalization()(up_conv2)
        up_conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(up_conv2)
        up_conv2 = BatchNormalization()(up_conv2)

        conv_t1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(up_conv2)
        conc1 = concatenate([conv_t1, conv1])
        up_conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(conc1)
        up_conv1 = BatchNormalization()(up_conv1)
        up_conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(up_conv1)
        up_conv1 = BatchNormalization()(up_conv1)

        output_layer = Conv2D(2, (1, 1), padding="same", activation="sigmoid")(up_conv1)

        self.model = tf.keras.Model(inputs=inputs_layer, outputs=output_layer)

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_function,
                           metrics=get_metrics())

    def unet_light(self, dim, input_channels, batch_size):

        """
        Unet implementation:
        - https://arxiv.org/abs/1505.04597

        :param dim: dimension of inputs
        :param input_channels: number of bands/layers of input
        :param batch_size: # batches in the input
        :return: compiled model
        """

        inputs_layer = tf.keras.layers.Input((dim[0], dim[1], input_channels), batch_size=batch_size)
        conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(inputs_layer)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D((2, 2))(conv1)

        conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D((2, 2))(conv2)

        conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D((2, 2))(conv3)

        conv_middle = Conv2D(512, (3, 3), activation="relu", padding="same")(pool3)
        conv_middle = BatchNormalization()(conv_middle)
        conv_middle = Conv2D(512, (3, 3), activation="relu", padding="same")(conv_middle)
        conv_middle = BatchNormalization()(conv_middle)

        conv_t3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(conv_middle)
        conc3 = concatenate([conv_t3, conv3])
        up_conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(conc3)
        up_conv3 = BatchNormalization()(up_conv3)
        up_conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(up_conv3)
        up_conv3 = BatchNormalization()(up_conv3)

        conv_t2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(up_conv3)
        conc2 = concatenate([conv_t2, conv2])
        up_conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(conc2)
        up_conv2 = BatchNormalization()(up_conv2)
        up_conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(up_conv2)
        up_conv2 = BatchNormalization()(up_conv2)

        conv_t1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(up_conv2)
        conc1 = concatenate([conv_t1, conv1])
        up_conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(conc1)
        up_conv1 = BatchNormalization()(up_conv1)
        up_conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(up_conv1)
        up_conv1 = BatchNormalization()(up_conv1)

        output_layer = Conv2D(2, (1, 1), padding="same", activation="sigmoid")(up_conv1)

        self.model = tf.keras.Model(inputs=inputs_layer, outputs=output_layer)

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_function,
                           metrics=get_metrics())

    def srcnn_unet(self, dim, input_channels, batch_size, srcnn_count):

        """
        Stacked SRCNN+Unet implementation. Resources:
        - https://arxiv.org/abs/1501.00092
        - https://arxiv.org/abs/1505.04597

        :param dim: dimension of inputs
        :param input_channels: number of bands/layers of input
        :param batch_size: # batches in the input
        :param srcnn_count: number of bands/layers that SRCNN will be applied.
        :return: compiled model
        """

        input_layers = []
        input_conc_layers = []
        for i in range(0, srcnn_count):
            temp_input_layer = tf.keras.layers.Input((dim[0], dim[1], 1), batch_size=batch_size)
            input_layers.append(temp_input_layer)
            srcnn = Conv2D(64, (9, 9), activation='relu', padding="same")(temp_input_layer)
            srcnn = Conv2D(32, (1, 1), activation='relu', padding="same")(srcnn)
            srcnn = Conv2D(1, (5, 5), activation='relu', padding="same")(srcnn)
            input_conc_layers.append(srcnn)

        if input_channels - srcnn_count > 0:
            non_srcnn_layers = tf.keras.layers.Input((dim[0], dim[1], 3), batch_size=batch_size)
            input_layers.append(non_srcnn_layers)
            input_conc_layers.append(non_srcnn_layers)

        main_inputs = concatenate(input_conc_layers)

        # TODO: replace code with Unet method to remove code duplication

        conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(main_inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D((2, 2))(conv1)

        conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D((2, 2))(conv2)

        conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D((2, 2))(conv3)

        conv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv4)
        conv4 = BatchNormalization()(conv4)
        pool4 = MaxPooling2D((2, 2))(conv4)

        conv_middle = Conv2D(1024, (3, 3), activation="relu", padding="same")(pool4)
        conv_middle = BatchNormalization()(conv_middle)
        conv_middle = Conv2D(1024, (3, 3), activation="relu", padding="same")(conv_middle)
        conv_middle = BatchNormalization()(conv_middle)

        conv_t4 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(conv_middle)
        conc4 = concatenate([conv_t4, conv4])
        up_conv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(conc4)
        up_conv4 = BatchNormalization()(up_conv4)
        up_conv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(up_conv4)
        up_conv4 = BatchNormalization()(up_conv4)

        conv_t3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(up_conv4)
        conc3 = concatenate([conv_t3, conv3])
        up_conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(conc3)
        up_conv3 = BatchNormalization()(up_conv3)
        up_conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(up_conv3)
        up_conv3 = BatchNormalization()(up_conv3)

        conv_t2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(up_conv3)
        conc2 = concatenate([conv_t2, conv2])
        up_conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(conc2)
        up_conv2 = BatchNormalization()(up_conv2)
        up_conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(up_conv2)
        up_conv2 = BatchNormalization()(up_conv2)

        conv_t1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(up_conv2)
        conc1 = concatenate([conv_t1, conv1])
        up_conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(conc1)
        up_conv1 = BatchNormalization()(up_conv1)
        up_conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(up_conv1)
        up_conv1 = BatchNormalization()(up_conv1)

        output_layer = Conv2D(2, (1, 1), padding="same", activation="sigmoid")(up_conv1)

        self.model = tf.keras.Model(inputs=input_layers, outputs=output_layer)

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_function,
                           metrics=get_metrics())

    def resunet(self, dim, input_channels, batch_size):

        """
        ResUnet implemented using
        - https://arxiv.org/pdf/1711.10684.pdf
        - https://github.com/nikhilroxtomar/Deep-Residual-Unet/blob/master/Deep%20Residual%20UNet.ipynb

        :param dim: dimension of inputs
        :param input_channels: number of bands/layers of input
        :param batch_size: # batches in the input
        :return: compiled model
        """

        inputs_layer = tf.keras.layers.Input((dim[0], dim[1], input_channels), batch_size=batch_size)

        conv1 = Conv2D(64, (3, 3), padding="same", strides=1)(inputs_layer)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation("relu")(conv1)
        conv1 = Conv2D(64, (3, 3), padding="same", strides=1)(conv1)
        conv1 = BatchNormalization()(conv1)

        res1 = Conv2D(64, (1, 1), padding="same", strides=1)(inputs_layer)
        res1 = BatchNormalization()(res1)

        conv1_output = Add()([conv1, res1])
        conv1_output = Activation("relu")(conv1_output)

        conv2 = Conv2D(128, (3, 3), padding="same", strides=2)(conv1_output)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation("relu")(conv2)
        conv2 = Conv2D(128, (3, 3), padding="same", strides=1)(conv2)
        conv2 = BatchNormalization()(conv2)

        res2 = Conv2D(128, (1, 1), padding="same", strides=2)(conv1_output)
        res2 = BatchNormalization()(res2)

        conv2_output = Add()([conv2, res2])
        conv2_output = Activation("relu")(conv2_output)

        conv3 = Conv2D(256, (3, 3), padding="same", strides=2)(conv2_output)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation("relu")(conv3)
        conv3 = Conv2D(256, (3, 3), padding="same", strides=1)(conv3)
        conv3 = BatchNormalization()(conv3)

        res3 = Conv2D(256, (1, 1), padding="same", strides=2)(conv2_output)
        res3 = BatchNormalization()(res3)

        conv3_output = Add()([conv3, res3])
        conv3_output = Activation("relu")(conv3_output)

        conv4 = Conv2D(512, (3, 3), padding="same", strides=2)(conv3_output)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation("relu")(conv4)
        conv4 = Conv2D(512, (3, 3), padding="same", strides=1)(conv4)
        conv4 = BatchNormalization()(conv4)

        res4 = Conv2D(512, (1, 1), padding="same", strides=2)(conv3_output)
        res4 = BatchNormalization()(res4)

        conv4_output = Add()([conv4, res4])
        conv4_output = Activation("relu")(conv4_output)

        conv_middle = Conv2D(1024, (3, 3), padding="same", strides=2)(conv4_output)
        conv_middle = BatchNormalization()(conv_middle)
        conv_middle = Activation("relu")(conv_middle)
        conv_middle = Conv2D(1024, (3, 3), padding="same", strides=1)(conv_middle)
        conv_middle = BatchNormalization()(conv_middle)

        res_middle = Conv2D(1024, (1, 1), padding="same", strides=2)(conv4_output)
        res_middle = BatchNormalization()(res_middle)

        conv_middle_output = Add()([conv_middle, res_middle])
        conv_middle_output = Activation("relu")(conv_middle_output)

        upconv1_trans = Conv2DTranspose(512, (3, 3), padding="same", strides=2)(conv_middle_output)
        conc1 = concatenate([upconv1_trans, conv4_output])

        upconv1 = Conv2D(512, (3, 3), padding="same", strides=1)(conc1)
        upconv1 = BatchNormalization()(upconv1)
        upconv1 = Activation("relu")(upconv1)
        upconv1 = Conv2D(512, (3, 3), padding="same", strides=1)(upconv1)
        upconv1 = BatchNormalization()(upconv1)

        res_up_1 = Conv2D(512, (1, 1), padding="same", strides=1)(conc1)
        res_up_1 = BatchNormalization()(res_up_1)

        upconv1_output = Add()([upconv1, res_up_1])
        upconv1_output = Activation("relu")(upconv1_output)

        upconv2_trans = Conv2DTranspose(256, (3, 3), padding="same", strides=2)(upconv1_output)
        conc2 = concatenate([upconv2_trans, conv3_output])

        upconv2 = Conv2D(256, (3, 3), padding="same", strides=1)(conc2)
        upconv2 = BatchNormalization()(upconv2)
        upconv2 = Activation("relu")(upconv2)
        upconv2 = Conv2D(256, (3, 3), padding="same", strides=1)(upconv2)
        upconv2 = BatchNormalization()(upconv2)

        res_up_2 = Conv2D(256, (1, 1), padding="same", strides=1)(conc2)
        res_up_2 = BatchNormalization()(res_up_2)

        upconv2_output = Add()([upconv2, res_up_2])
        upconv2_output = Activation("relu")(upconv2_output)

        upconv3_trans = Conv2DTranspose(128, (3, 3), padding="same", strides=2)(upconv2_output)
        conc3 = concatenate([upconv3_trans, conv2_output])

        upconv3 = Conv2D(128, (3, 3), padding="same", strides=1)(conc3)
        upconv3 = BatchNormalization()(upconv3)
        upconv3 = Activation("relu")(upconv3)
        upconv3 = Conv2D(128, (3, 3), padding="same", strides=1)(upconv3)
        upconv3 = BatchNormalization()(upconv3)

        res_up_3 = Conv2D(128, (1, 1), padding="same", strides=1)(conc3)
        res_up_3 = BatchNormalization()(res_up_3)

        upconv3_output = Add()([upconv3, res_up_3])
        upconv3_output = Activation("relu")(upconv3_output)

        upconv4_trans = Conv2DTranspose(64, (3, 3), padding="same", strides=2)(upconv3_output)
        conc4 = concatenate([upconv4_trans, conv1_output])

        upconv4 = Conv2D(64, (3, 3), padding="same", strides=1)(conc4)
        upconv4 = BatchNormalization()(upconv4)
        upconv4 = Activation("relu")(upconv4)
        upconv4 = Conv2D(64, (3, 3), padding="same", strides=1)(upconv4)
        upconv4 = BatchNormalization()(upconv4)

        res_up_4 = Conv2D(64, (1, 1), padding="same", strides=1)(conc4)
        res_up_4 = BatchNormalization()(res_up_4)

        upconv4_output = Add()([upconv4, res_up_4])
        upconv4_output = Activation("relu")(upconv4_output)

        output_layer = Conv2D(2, (1, 1), padding="same", activation="sigmoid")(upconv4_output)

        self.model = tf.keras.Model(inputs=inputs_layer, outputs=output_layer)

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_function,
                           metrics=get_metrics())

    def resunet_light(self, dim, input_channels, batch_size):

        """
        ResUnet implemented using
        - https://arxiv.org/pdf/1711.10684.pdf
        - https://github.com/nikhilroxtomar/Deep-Residual-Unet/blob/master/Deep%20Residual%20UNet.ipynb

        :param dim: dimension of inputs
        :param input_channels: number of bands/layers of input
        :param batch_size: # batches in the input
        :return: compiled model
        """

        inputs_layer = tf.keras.layers.Input((dim[0], dim[1], input_channels), batch_size=batch_size)

        conv1 = Conv2D(64, (3, 3), padding="same", strides=1)(inputs_layer)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation("relu")(conv1)
        conv1 = Conv2D(64, (3, 3), padding="same", strides=1)(conv1)
        conv1 = BatchNormalization()(conv1)

        res1 = Conv2D(64, (1, 1), padding="same", strides=1)(inputs_layer)
        res1 = BatchNormalization()(res1)

        conv1_output = Add()([conv1, res1])
        conv1_output = Activation("relu")(conv1_output)

        conv2 = Conv2D(128, (3, 3), padding="same", strides=2)(conv1_output)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation("relu")(conv2)
        conv2 = Conv2D(128, (3, 3), padding="same", strides=1)(conv2)
        conv2 = BatchNormalization()(conv2)

        res2 = Conv2D(128, (1, 1), padding="same", strides=2)(conv1_output)
        res2 = BatchNormalization()(res2)

        conv2_output = Add()([conv2, res2])
        conv2_output = Activation("relu")(conv2_output)

        conv3 = Conv2D(256, (3, 3), padding="same", strides=2)(conv2_output)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation("relu")(conv3)
        conv3 = Conv2D(256, (3, 3), padding="same", strides=1)(conv3)
        conv3 = BatchNormalization()(conv3)

        res3 = Conv2D(256, (1, 1), padding="same", strides=2)(conv2_output)
        res3 = BatchNormalization()(res3)

        conv3_output = Add()([conv3, res3])
        conv3_output = Activation("relu")(conv3_output)

        conv_middle = Conv2D(512, (3, 3), padding="same", strides=2)(conv3_output)
        conv_middle = BatchNormalization()(conv_middle)
        conv_middle = Activation("relu")(conv_middle)
        conv_middle = Conv2D(512, (3, 3), padding="same", strides=1)(conv_middle)
        conv_middle = BatchNormalization()(conv_middle)

        res_middle = Conv2D(512, (1, 1), padding="same", strides=2)(conv3_output)
        res_middle = BatchNormalization()(res_middle)

        conv_middle_output = Add()([conv_middle, res_middle])
        conv_middle_output = Activation("relu")(conv_middle_output)

        upconv2_trans = Conv2DTranspose(256, (3, 3), padding="same", strides=2)(conv_middle_output)
        conc2 = concatenate([upconv2_trans, conv3_output])

        upconv2 = Conv2D(256, (3, 3), padding="same", strides=1)(conc2)
        upconv2 = BatchNormalization()(upconv2)
        upconv2 = Activation("relu")(upconv2)
        upconv2 = Conv2D(256, (3, 3), padding="same", strides=1)(upconv2)
        upconv2 = BatchNormalization()(upconv2)

        res_up_2 = Conv2D(256, (1, 1), padding="same", strides=1)(conc2)
        res_up_2 = BatchNormalization()(res_up_2)

        upconv2_output = Add()([upconv2, res_up_2])
        upconv2_output = Activation("relu")(upconv2_output)

        upconv3_trans = Conv2DTranspose(128, (3, 3), padding="same", strides=2)(upconv2_output)
        conc3 = concatenate([upconv3_trans, conv2_output])

        upconv3 = Conv2D(128, (3, 3), padding="same", strides=1)(conc3)
        upconv3 = BatchNormalization()(upconv3)
        upconv3 = Activation("relu")(upconv3)
        upconv3 = Conv2D(128, (3, 3), padding="same", strides=1)(upconv3)
        upconv3 = BatchNormalization()(upconv3)

        res_up_3 = Conv2D(128, (1, 1), padding="same", strides=1)(conc3)
        res_up_3 = BatchNormalization()(res_up_3)

        upconv3_output = Add()([upconv3, res_up_3])
        upconv3_output = Activation("relu")(upconv3_output)

        upconv4_trans = Conv2DTranspose(64, (3, 3), padding="same", strides=2)(upconv3_output)
        conc4 = concatenate([upconv4_trans, conv1_output])

        upconv4 = Conv2D(64, (3, 3), padding="same", strides=1)(conc4)
        upconv4 = BatchNormalization()(upconv4)
        upconv4 = Activation("relu")(upconv4)
        upconv4 = Conv2D(64, (3, 3), padding="same", strides=1)(upconv4)
        upconv4 = BatchNormalization()(upconv4)

        res_up_4 = Conv2D(64, (1, 1), padding="same", strides=1)(conc4)
        res_up_4 = BatchNormalization()(res_up_4)

        upconv4_output = Add()([upconv4, res_up_4])
        upconv4_output = Activation("relu")(upconv4_output)

        output_layer = Conv2D(2, (1, 1), padding="same", activation="sigmoid")(upconv4_output)

        self.model = tf.keras.Model(inputs=inputs_layer, outputs=output_layer)

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_function,
                           metrics=get_metrics())

    def dlinknet(self, dim, input_channels, batch_size):

        '''
        D-LinkNet implementation. Derived from:
        - D-LinkNet: https://arxiv.org/abs/1707.03718
        - D-LinkNet Pytorch implementation: https://github.com/zlkanata/DeepGlobe-Road-Extraction-Challenge/blob/7354f87ea03224a2c7a4c9e2bc6988cb511eb9a8/networks/dinknet.py
        - LeNet: https://arxiv.org/abs/1707.03718
        - ResNet34: https://arxiv.org/abs/1512.03385
        - ResNet34 implementation guidance: https://towardsdatascience.com/understand-and-implement-resnet-50-with-tensorflow-2-0-1190b9b52691

        :param dim:
        :param input_channels:
        :param batch_size:
        :return:
        '''

        inputs_layer = tf.keras.layers.Input((dim[0], dim[1], input_channels), batch_size=batch_size)

        conv1 = Conv2D(64, (7, 7), padding="same", strides=2)(inputs_layer)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation("relu")(conv1)
        conv1 = MaxPooling2D((3, 3), strides=2, padding="same")(conv1)

        # level1
        conv2 = Conv2D(64, (3, 3), padding="same", strides=1)(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation("relu")(conv2)
        conv2 = Conv2D(64, (3, 3), padding="same", strides=1)(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation("relu")(conv2)
        conv2_add = Add()([conv2, conv1])
        conv2_add = BatchNormalization()(conv2_add)
        conv2_add = Activation("relu")(conv2_add)

        conv3 = Conv2D(64, (3, 3), padding="same", strides=1)(conv2_add)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation("relu")(conv3)
        conv3 = Conv2D(64, (3, 3), padding="same", strides=1)(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation("relu")(conv3)
        conv3_add = Add()([conv3, conv2_add])
        conv3_add = BatchNormalization()(conv3_add)
        conv3_add = Activation("relu")(conv3_add)

        conv4 = Conv2D(64, (3, 3), padding="same", strides=1)(conv3_add)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation("relu")(conv4)
        conv4 = Conv2D(64, (3, 3), padding="same", strides=1)(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation("relu")(conv4)
        conv4_add = Add()([conv4, conv3_add])
        conv4_add = BatchNormalization()(conv4_add)
        conv4_add = Activation("relu")(conv4_add)

        # level2
        conv5 = Conv2D(128, (3, 3), padding="same", strides=2)(conv4_add)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation("relu")(conv5)
        conv5 = Conv2D(128, (3, 3), padding="same", strides=1)(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation("relu")(conv5)
        conv4_add_skip = Conv2D(128, (1, 1), padding="same", strides=2)(conv4_add)
        conv5_add = Add()([conv5, conv4_add_skip])
        conv5_add = BatchNormalization()(conv5_add)
        conv5_add = Activation("relu")(conv5_add)

        conv6 = Conv2D(128, (3, 3), padding="same", strides=1)(conv5_add)
        conv6 = BatchNormalization()(conv6)
        conv6 = Activation("relu")(conv6)
        conv6 = Conv2D(128, (3, 3), padding="same", strides=1)(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Activation("relu")(conv6)
        conv6_add = Add()([conv6, conv5_add])
        conv6_add = BatchNormalization()(conv6_add)
        conv6_add = Activation("relu")(conv6_add)

        conv7 = Conv2D(128, (3, 3), padding="same", strides=1)(conv6_add)
        conv7 = BatchNormalization()(conv7)
        conv7 = Activation("relu")(conv7)
        conv7 = Conv2D(128, (3, 3), padding="same", strides=1)(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Activation("relu")(conv7)
        conv7_add = Add()([conv7, conv6_add])
        conv7_add = BatchNormalization()(conv7_add)
        conv7_add = Activation("relu")(conv7_add)

        conv8 = Conv2D(128, (3, 3), padding="same", strides=1)(conv7_add)
        conv8 = BatchNormalization()(conv8)
        conv8 = Activation("relu")(conv8)
        conv8 = Conv2D(128, (3, 3), padding="same", strides=1)(conv8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Activation("relu")(conv8)
        conv8_add = Add()([conv8, conv7_add])
        conv8_add = BatchNormalization()(conv8_add)
        conv8_add = Activation("relu")(conv8_add)

        # level3
        conv9 = Conv2D(256, (3, 3), padding="same", strides=2)(conv8_add)
        conv9 = BatchNormalization()(conv9)
        conv9 = Activation("relu")(conv9)
        conv9 = Conv2D(256, (3, 3), padding="same", strides=1)(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Activation("relu")(conv9)
        conv8_add_skip = Conv2D(256, (1, 1), padding="same", strides=2)(conv8_add)
        conv9_add = Add()([conv9, conv8_add_skip])
        conv9_add = BatchNormalization()(conv9_add)
        conv9_add = Activation("relu")(conv9_add)

        conv10 = Conv2D(256, (3, 3), padding="same", strides=1)(conv9_add)
        conv10 = BatchNormalization()(conv10)
        conv10 = Activation("relu")(conv10)
        conv10 = Conv2D(256, (3, 3), padding="same", strides=1)(conv10)
        conv10 = BatchNormalization()(conv10)
        conv10 = Activation("relu")(conv10)
        conv10_add = Add()([conv10, conv9_add])
        conv10_add = BatchNormalization()(conv10_add)
        conv10_add = Activation("relu")(conv10_add)

        conv11 = Conv2D(256, (3, 3), padding="same", strides=1)(conv10_add)
        conv11 = BatchNormalization()(conv11)
        conv11 = Activation("relu")(conv11)
        conv11 = Conv2D(256, (3, 3), padding="same", strides=1)(conv11)
        conv11 = BatchNormalization()(conv11)
        conv11 = Activation("relu")(conv11)
        conv11_add = Add()([conv11, conv10_add])
        conv11_add = BatchNormalization()(conv11_add)
        conv11_add = Activation("relu")(conv11_add)

        conv12 = Conv2D(256, (3, 3), padding="same", strides=1)(conv11_add)
        conv12 = BatchNormalization()(conv12)
        conv12 = Activation("relu")(conv12)
        conv12 = Conv2D(256, (3, 3), padding="same", strides=1)(conv12)
        conv12 = BatchNormalization()(conv12)
        conv12 = Activation("relu")(conv12)
        conv12_add = Add()([conv12, conv11_add])
        conv12_add = BatchNormalization()(conv12_add)
        conv12_add = Activation("relu")(conv12_add)

        conv13 = Conv2D(256, (3, 3), padding="same", strides=1)(conv12_add)
        conv13 = BatchNormalization()(conv13)
        conv13 = Activation("relu")(conv13)
        conv13 = Conv2D(256, (3, 3), padding="same", strides=1)(conv13)
        conv13 = BatchNormalization()(conv13)
        conv13 = Activation("relu")(conv13)
        conv13_add = Add()([conv13, conv12_add])
        conv13_add = BatchNormalization()(conv13_add)
        conv13_add = Activation("relu")(conv13_add)

        conv14 = Conv2D(256, (3, 3), padding="same", strides=1)(conv13_add)
        conv14 = BatchNormalization()(conv14)
        conv14 = Activation("relu")(conv14)
        conv14 = Conv2D(256, (3, 3), padding="same", strides=1)(conv14)
        conv14 = BatchNormalization()(conv14)
        conv14 = Activation("relu")(conv14)
        conv14_add = Add()([conv14, conv13_add])
        conv14_add = BatchNormalization()(conv14_add)
        conv14_add = Activation("relu")(conv14_add)

        # level4
        conv15 = Conv2D(512, (3, 3), padding="same", strides=2)(conv14_add)
        conv15 = BatchNormalization()(conv15)
        conv15 = Activation("relu")(conv15)
        conv15 = Conv2D(512, (3, 3), padding="same", strides=1)(conv15)
        conv15 = BatchNormalization()(conv15)
        conv15 = Activation("relu")(conv15)
        conv14_add_skip = Conv2D(512, (1, 1), padding="same", strides=2)(conv14_add)
        conv15_add = Add()([conv15, conv14_add_skip])
        conv15_add = BatchNormalization()(conv15_add)
        conv15_add = Activation("relu")(conv15_add)

        conv16 = Conv2D(512, (3, 3), padding="same", strides=1)(conv15_add)
        conv16 = BatchNormalization()(conv16)
        conv16 = Activation("relu")(conv16)
        conv16 = Conv2D(512, (3, 3), padding="same", strides=1)(conv16)
        conv16 = BatchNormalization()(conv16)
        conv16 = Activation("relu")(conv16)
        conv16_add = Add()([conv16, conv15_add])
        conv16_add = BatchNormalization()(conv16_add)
        conv16_add = Activation("relu")(conv16_add)

        conv17 = Conv2D(512, (3, 3), padding="same", strides=1)(conv16_add)
        conv17 = BatchNormalization()(conv17)
        conv17 = Activation("relu")(conv17)
        conv17 = Conv2D(512, (3, 3), padding="same", strides=1)(conv17)
        conv17 = BatchNormalization()(conv17)
        conv17 = Activation("relu")(conv17)
        conv17_add = Add()([conv17, conv16_add])
        conv17_add = BatchNormalization()(conv17_add)
        conv17_add = Activation("relu")(conv17_add)

        last_pool = AveragePooling2D((2, 2), padding="same", strides=1)(conv17_add)

        # dilation8
        dilation8 = Conv2D(512, (3, 3), dilation_rate=1, padding="same")(last_pool)
        dilation8 = Activation("relu")(dilation8)
        dilation8 = Conv2D(512, (3, 3), dilation_rate=2, padding="same")(dilation8)
        dilation8 = Activation("relu")(dilation8)
        dilation8 = Conv2D(512, (3, 3), dilation_rate=4, padding="same")(dilation8)
        dilation8 = Activation("relu")(dilation8)
        dilation8 = Conv2D(512, (3, 3), dilation_rate=8, padding="same")(dilation8)
        dilation8 = Activation("relu")(dilation8)

        # dilation4
        dilation4 = Conv2D(512, (3, 3), dilation_rate=1, padding="same")(last_pool)
        dilation4 = Activation("relu")(dilation4)
        dilation4 = Conv2D(512, (3, 3), dilation_rate=2, padding="same")(dilation4)
        dilation4 = Activation("relu")(dilation4)
        dilation4 = Conv2D(512, (3, 3), dilation_rate=4, padding="same")(dilation4)
        dilation4 = Activation("relu")(dilation4)

        # dilation2
        dilation2 = Conv2D(512, (3, 3), dilation_rate=1, padding="same")(last_pool)
        dilation2 = Activation("relu")(dilation2)
        dilation2 = Conv2D(512, (3, 3), dilation_rate=2, padding="same")(dilation2)
        dilation2 = Activation("relu")(dilation2)

        # dilation1
        dilation1 = Conv2D(512, (3, 3), dilation_rate=1, padding="same")(last_pool)
        dilation1 = Activation("relu")(dilation1)

        dilation_out = Add()([last_pool, dilation1, dilation2, dilation4, dilation8])

        # decode level3
        decode3 = Conv2D(512, (1, 1), padding="same", strides=1)(dilation_out)
        decode3 = BatchNormalization()(decode3)
        decode3 = Activation("relu")(decode3)

        decode3 = Conv2DTranspose(128, (3, 3), strides=2, padding="same")(decode3)
        decode3 = BatchNormalization()(decode3)
        decode3 = Activation("relu")(decode3)

        decode3 = Conv2D(256, (1, 1), strides=1, padding="same")(decode3)
        decode3 = BatchNormalization()(decode3)
        decode3 = Activation("relu")(decode3)

        decode3_out = Add()([decode3, conv14_add])

        # decode level2
        decode2 = Conv2D(256, (1, 1), padding="same", strides=1)(decode3_out)
        decode2 = BatchNormalization()(decode2)
        decode2 = Activation("relu")(decode2)

        decode2 = Conv2DTranspose(64, (3, 3), strides=2, padding="same")(decode2)
        decode2 = BatchNormalization()(decode2)
        decode2 = Activation("relu")(decode2)

        decode2 = Conv2D(128, (1, 1), strides=1, padding="same")(decode2)
        decode2 = BatchNormalization()(decode2)
        decode2 = Activation("relu")(decode2)

        decode2_out = Add()([decode2, conv8_add])

        # decode level1
        decode1 = Conv2D(128, (1, 1), padding="same", strides=1)(decode2_out)
        decode1 = BatchNormalization()(decode1)
        decode1 = Activation("relu")(decode1)

        decode1 = Conv2DTranspose(32, (3, 3), strides=2, padding="same")(decode1)
        decode1 = BatchNormalization()(decode1)
        decode1 = Activation("relu")(decode1)

        decode1 = Conv2D(64, (1, 1), strides=1, padding="same")(decode1)
        decode1 = BatchNormalization()(decode1)
        decode1 = Activation("relu")(decode1)

        # decode out
        decode_out = Add()([decode1, conv4_add])

        final_conv = Conv2DTranspose(64, (3, 3), padding="same", strides=2)(decode_out)
        final_conv = Activation("relu")(final_conv)
        final_conv = Conv2D(32, (3, 3), padding="same")(final_conv)
        final_conv = Activation("relu")(final_conv)

        output_layer = Conv2DTranspose(2, (3, 3), padding="same", activation="sigmoid", strides=2)(final_conv)

        self.model = tf.keras.Model(inputs=inputs_layer, outputs=output_layer)

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_function,
                           metrics=get_metrics())