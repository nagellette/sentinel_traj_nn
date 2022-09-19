import sys

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, \
    Activation, Add
from tensorflow.python.keras.layers import AveragePooling2D

from utils.construct_loss_function import ConstructLossFunction
from utils.construct_optimizer import ConstructOptimizer
from utils.get_metrics import get_metrics
from utils.get_fusion_layer import get_fusion_layer


# noinspection DuplicatedCode
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
                 loss="dice",
                 fusion_type="average"):
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
        :param fusion_type: fusion type to be used in satellite+trajectory late fusion models. Set to default for early
        fusion examples and not being used.
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
        self.fusion_type = fusion_type

        self.loss_function = ConstructLossFunction(loss_function_name=self.loss,
                                                   batch_size=self.batch_size).get_loss_function()
        self.optimizer = ConstructOptimizer(optimizer_name=self.optimizer, l_rate=self.l_rate, decay=self.decay,
                                            momentum=self.momentum, nesterov=self.nesterov).get_optimizer()

        # model to run
        if self.model_name == "test_model":
            self.test_model()
        elif self.model_name == "unet":
            self.unet(self.dim, self.input_channels, self.batch_size)
        elif self.model_name == "unetlight":
            self.unet_light(self.dim, self.input_channels, self.batch_size)
        elif self.model_name == "unet_traj_type1":
            self.unet_traj_type1(self.dim, self.input_channels, self.batch_size, self.fusion_type)
        elif self.model_name == "unet_traj_type2":
            self.unet_traj_type2(self.dim, self.input_channels, self.batch_size, self.fusion_type)
        elif self.model_name == "srcnn_unet":
            self.srcnn_unet(self.dim, self.input_channels, self.batch_size, self.srcnn_count)
        elif self.model_name == "resunet":
            self.resunet(self.dim, self.input_channels, self.batch_size)
        elif self.model_name == "resunetlight":
            self.resunet_light(self.dim, self.input_channels, self.batch_size)
        elif self.model_name == "resunet_traj_type1":
            self.resunet_traj_type1(self.dim, self.input_channels, self.batch_size, self.fusion_type)
        elif self.model_name == "resunet_traj_type2":
            self.resunet_traj_type2(self.dim, self.input_channels, self.batch_size, self.fusion_type)
        elif self.model_name == "dlinknet":
            self.dlinknet(self.dim, self.input_channels, self.batch_size)
        elif self.model_name == "dlinknet_traj_type1":
            self.dlinknet_traj_type1(self.dim, self.input_channels, self.batch_size)
        elif self.model_name == "dlinknet_traj_type2":
            self.dlinknet_traj_type2(self.dim, self.input_channels, self.batch_size)
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
                           metrics=get_metrics(batch_size=self.batch_size))

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
                           metrics=get_metrics(batch_size=self.batch_size))

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
                           metrics=get_metrics(batch_size=self.batch_size))

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
                           metrics=get_metrics(batch_size=self.batch_size))

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
                           metrics=get_metrics(batch_size=self.batch_size))

    def dlinknet(self, dim, input_channels, batch_size):

        """

        D-LinkNet implementation. Derived from: - D-LinkNet: https://arxiv.org/abs/1707.03718 - D-LinkNet Pytorch
        implementation: https://github.com/zlkanata/DeepGlobe-Road-Extraction-Challenge/blob
        /7354f87ea03224a2c7a4c9e2bc6988cb511eb9a8/networks/dinknet.py - LeNet: https://arxiv.org/abs/1707.03718 -
        ResNet34: https://arxiv.org/abs/1512.03385 - ResNet34 implementation guidance:
        https://towardsdatascience.com/understand-and-implement-resnet-50-with-tensorflow-2-0-1190b9b52691

        :param dim: dimension of inputs
        :param input_channels: number of bands/layers of input
        :param batch_size: # batches in the input
        :return: compiled model
        """

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
                           metrics=get_metrics(batch_size=self.batch_size))

    def unet_traj_type1(self, dim, input_channels, batch_size, fusion_type):

        """
        Unet trajectory late fusion implementation with unet stream for satellite and direct connection to trajectory.
        CAUTION: This model works only with satellite and trajectory data and automatically pick last array as
        trajectory array.

        :param dim: dimension of inputs
        :param input_channels: number of bands/layers of input
        :param batch_size: # batches in the input
        :param fusion_type: preferred fusion type to be used
        :return: compiled model
        """

        inputs_layer = tf.keras.layers.Input((dim[0], dim[1], input_channels), batch_size=batch_size)

        # split input into satellite and trajectory tensors
        input_sat, input_traj = tf.split(inputs_layer, [input_channels - 1, 1], axis=3)

        # satellite unet stream
        conv1_sat = Conv2D(64, (3, 3), activation="relu", padding="same")(input_sat)
        conv1_sat = BatchNormalization()(conv1_sat)
        conv1_sat = Conv2D(64, (3, 3), activation="relu", padding="same")(conv1_sat)
        conv1_sat = BatchNormalization()(conv1_sat)
        pool1_sat = MaxPooling2D((2, 2))(conv1_sat)

        conv2_sat = Conv2D(128, (3, 3), activation="relu", padding="same")(pool1_sat)
        conv2_sat = BatchNormalization()(conv2_sat)
        conv2_sat = Conv2D(128, (3, 3), activation="relu", padding="same")(conv2_sat)
        conv2_sat = BatchNormalization()(conv2_sat)
        pool2_sat = MaxPooling2D((2, 2))(conv2_sat)

        conv3_sat = Conv2D(256, (3, 3), activation="relu", padding="same")(pool2_sat)
        conv3_sat = BatchNormalization()(conv3_sat)
        conv3_sat = Conv2D(256, (3, 3), activation="relu", padding="same")(conv3_sat)
        conv3_sat = BatchNormalization()(conv3_sat)
        pool3_sat = MaxPooling2D((2, 2))(conv3_sat)

        conv4_sat = Conv2D(512, (3, 3), activation="relu", padding="same")(pool3_sat)
        conv4_sat = BatchNormalization()(conv4_sat)
        conv4_sat = Conv2D(512, (3, 3), activation="relu", padding="same")(conv4_sat)
        conv4_sat = BatchNormalization()(conv4_sat)
        pool4_sat = MaxPooling2D((2, 2))(conv4_sat)

        conv_middle_sat = Conv2D(1024, (3, 3), activation="relu", padding="same")(pool4_sat)
        conv_middle_sat = BatchNormalization()(conv_middle_sat)
        conv_middle_sat = Conv2D(1024, (3, 3), activation="relu", padding="same")(conv_middle_sat)
        conv_middle_sat = BatchNormalization()(conv_middle_sat)

        conv_t4_sat = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(conv_middle_sat)
        conc4_sat = concatenate([conv_t4_sat, conv4_sat])
        up_conv4_sat = Conv2D(512, (3, 3), activation="relu", padding="same")(conc4_sat)
        up_conv4_sat = BatchNormalization()(up_conv4_sat)
        up_conv4_sat = Conv2D(512, (3, 3), activation="relu", padding="same")(up_conv4_sat)
        up_conv4_sat = BatchNormalization()(up_conv4_sat)

        conv_t3_sat = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(up_conv4_sat)
        conc3_sat = concatenate([conv_t3_sat, conv3_sat])
        up_conv3_sat = Conv2D(256, (3, 3), activation="relu", padding="same")(conc3_sat)
        up_conv3_sat = BatchNormalization()(up_conv3_sat)
        up_conv3_sat = Conv2D(256, (3, 3), activation="relu", padding="same")(up_conv3_sat)
        up_conv3_sat = BatchNormalization()(up_conv3_sat)

        conv_t2_sat = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(up_conv3_sat)
        conc2_sat = concatenate([conv_t2_sat, conv2_sat])
        up_conv2_sat = Conv2D(128, (3, 3), activation="relu", padding="same")(conc2_sat)
        up_conv2_sat = BatchNormalization()(up_conv2_sat)
        up_conv2_sat = Conv2D(128, (3, 3), activation="relu", padding="same")(up_conv2_sat)
        up_conv2_sat = BatchNormalization()(up_conv2_sat)

        conv_t1_sat = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(up_conv2_sat)
        conc1_sat = concatenate([conv_t1_sat, conv1_sat])
        up_conv1_sat = Conv2D(64, (3, 3), activation="relu", padding="same")(conc1_sat)
        up_conv1_sat = BatchNormalization()(up_conv1_sat)
        up_conv1_sat = Conv2D(64, (3, 3), activation="relu", padding="same")(up_conv1_sat)
        up_conv1_sat = BatchNormalization()(up_conv1_sat)

        output_layer1 = Conv2D(2, (1, 1), padding="same", activation="sigmoid")(up_conv1_sat)
        output_layer2 = Conv2D(2, (1, 1), padding="same", activation="sigmoid")(input_traj)

        fusion_layer = get_fusion_layer(fusion_type, output_layer1, output_layer2)

        if fusion_type == "concat":
            fusion_layer = Conv2D(2, (1, 1), padding="same", activation="sigmoid")(fusion_layer)

        self.model = tf.keras.Model(inputs=inputs_layer, outputs=fusion_layer)

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_function,
                           metrics=get_metrics(batch_size=self.batch_size))

    def unet_traj_type2(self, dim, input_channels, batch_size, fusion_type):

        """
        Unet trajectory late fusion implementation with unet stream for both satellite and trajectory.
        CAUTION: This model works only with satellite and trajectory data and automatically pick last array as
        trajectory array.

        :param dim: dimension of inputs
        :param input_channels: number of bands/layers of input
        :param batch_size: # batches in the input
        :param fusion_type: preferred fusion type to be used
        :return: compiled model
        """

        inputs_layer = tf.keras.layers.Input((dim[0], dim[1], input_channels), batch_size=batch_size)

        # split input into satellite and trajectory tensors
        input_sat, input_traj = tf.split(inputs_layer, [input_channels - 1, 1], axis=3)

        # satellite unet stream
        conv1_sat = Conv2D(64, (3, 3), activation="relu", padding="same")(input_sat)
        conv1_sat = BatchNormalization()(conv1_sat)
        conv1_sat = Conv2D(64, (3, 3), activation="relu", padding="same")(conv1_sat)
        conv1_sat = BatchNormalization()(conv1_sat)
        pool1_sat = MaxPooling2D((2, 2))(conv1_sat)

        conv2_sat = Conv2D(128, (3, 3), activation="relu", padding="same")(pool1_sat)
        conv2_sat = BatchNormalization()(conv2_sat)
        conv2_sat = Conv2D(128, (3, 3), activation="relu", padding="same")(conv2_sat)
        conv2_sat = BatchNormalization()(conv2_sat)
        pool2_sat = MaxPooling2D((2, 2))(conv2_sat)

        conv3_sat = Conv2D(256, (3, 3), activation="relu", padding="same")(pool2_sat)
        conv3_sat = BatchNormalization()(conv3_sat)
        conv3_sat = Conv2D(256, (3, 3), activation="relu", padding="same")(conv3_sat)
        conv3_sat = BatchNormalization()(conv3_sat)
        pool3_sat = MaxPooling2D((2, 2))(conv3_sat)

        conv4_sat = Conv2D(512, (3, 3), activation="relu", padding="same")(pool3_sat)
        conv4_sat = BatchNormalization()(conv4_sat)
        conv4_sat = Conv2D(512, (3, 3), activation="relu", padding="same")(conv4_sat)
        conv4_sat = BatchNormalization()(conv4_sat)
        pool4_sat = MaxPooling2D((2, 2))(conv4_sat)

        conv_middle_sat = Conv2D(1024, (3, 3), activation="relu", padding="same")(pool4_sat)
        conv_middle_sat = BatchNormalization()(conv_middle_sat)
        conv_middle_sat = Conv2D(1024, (3, 3), activation="relu", padding="same")(conv_middle_sat)
        conv_middle_sat = BatchNormalization()(conv_middle_sat)

        conv_t4_sat = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(conv_middle_sat)
        conc4_sat = concatenate([conv_t4_sat, conv4_sat])
        up_conv4_sat = Conv2D(512, (3, 3), activation="relu", padding="same")(conc4_sat)
        up_conv4_sat = BatchNormalization()(up_conv4_sat)
        up_conv4_sat = Conv2D(512, (3, 3), activation="relu", padding="same")(up_conv4_sat)
        up_conv4_sat = BatchNormalization()(up_conv4_sat)

        conv_t3_sat = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(up_conv4_sat)
        conc3_sat = concatenate([conv_t3_sat, conv3_sat])
        up_conv3_sat = Conv2D(256, (3, 3), activation="relu", padding="same")(conc3_sat)
        up_conv3_sat = BatchNormalization()(up_conv3_sat)
        up_conv3_sat = Conv2D(256, (3, 3), activation="relu", padding="same")(up_conv3_sat)
        up_conv3_sat = BatchNormalization()(up_conv3_sat)

        conv_t2_sat = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(up_conv3_sat)
        conc2_sat = concatenate([conv_t2_sat, conv2_sat])
        up_conv2_sat = Conv2D(128, (3, 3), activation="relu", padding="same")(conc2_sat)
        up_conv2_sat = BatchNormalization()(up_conv2_sat)
        up_conv2_sat = Conv2D(128, (3, 3), activation="relu", padding="same")(up_conv2_sat)
        up_conv2_sat = BatchNormalization()(up_conv2_sat)

        conv_t1_sat = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(up_conv2_sat)
        conc1_sat = concatenate([conv_t1_sat, conv1_sat])
        up_conv1_sat = Conv2D(64, (3, 3), activation="relu", padding="same")(conc1_sat)
        up_conv1_sat = BatchNormalization()(up_conv1_sat)
        up_conv1_sat = Conv2D(64, (3, 3), activation="relu", padding="same")(up_conv1_sat)
        up_conv1_sat = BatchNormalization()(up_conv1_sat)

        # trajectory unet stream
        conv1_traj = Conv2D(64, (3, 3), activation="relu", padding="same")(input_traj)
        conv1_traj = BatchNormalization()(conv1_traj)
        conv1_traj = Conv2D(64, (3, 3), activation="relu", padding="same")(conv1_traj)
        conv1_traj = BatchNormalization()(conv1_traj)
        pool1_traj = MaxPooling2D((2, 2))(conv1_traj)

        conv2_traj = Conv2D(128, (3, 3), activation="relu", padding="same")(pool1_traj)
        conv2_traj = BatchNormalization()(conv2_traj)
        conv2_traj = Conv2D(128, (3, 3), activation="relu", padding="same")(conv2_traj)
        conv2_traj = BatchNormalization()(conv2_traj)
        pool2_traj = MaxPooling2D((2, 2))(conv2_traj)

        conv3_traj = Conv2D(256, (3, 3), activation="relu", padding="same")(pool2_traj)
        conv3_traj = BatchNormalization()(conv3_traj)
        conv3_traj = Conv2D(256, (3, 3), activation="relu", padding="same")(conv3_traj)
        conv3_traj = BatchNormalization()(conv3_traj)
        pool3_traj = MaxPooling2D((2, 2))(conv3_traj)

        conv4_traj = Conv2D(512, (3, 3), activation="relu", padding="same")(pool3_traj)
        conv4_traj = BatchNormalization()(conv4_traj)
        conv4_traj = Conv2D(512, (3, 3), activation="relu", padding="same")(conv4_traj)
        conv4_traj = BatchNormalization()(conv4_traj)
        pool4_traj = MaxPooling2D((2, 2))(conv4_traj)

        conv_middle_traj = Conv2D(1024, (3, 3), activation="relu", padding="same")(pool4_traj)
        conv_middle_traj = BatchNormalization()(conv_middle_traj)
        conv_middle_traj = Conv2D(1024, (3, 3), activation="relu", padding="same")(conv_middle_traj)
        conv_middle_traj = BatchNormalization()(conv_middle_traj)

        conv_t4_traj = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(conv_middle_traj)
        conc4_traj = concatenate([conv_t4_traj, conv4_traj])
        up_conv4_traj = Conv2D(512, (3, 3), activation="relu", padding="same")(conc4_traj)
        up_conv4_traj = BatchNormalization()(up_conv4_traj)
        up_conv4_traj = Conv2D(512, (3, 3), activation="relu", padding="same")(up_conv4_traj)
        up_conv4_traj = BatchNormalization()(up_conv4_traj)

        conv_t3_traj = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(up_conv4_traj)
        conc3_traj = concatenate([conv_t3_traj, conv3_traj])
        up_conv3_traj = Conv2D(256, (3, 3), activation="relu", padding="same")(conc3_traj)
        up_conv3_traj = BatchNormalization()(up_conv3_traj)
        up_conv3_traj = Conv2D(256, (3, 3), activation="relu", padding="same")(up_conv3_traj)
        up_conv3_traj = BatchNormalization()(up_conv3_traj)

        conv_t2_traj = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(up_conv3_traj)
        conc2_traj = concatenate([conv_t2_traj, conv2_traj])
        up_conv2_traj = Conv2D(128, (3, 3), activation="relu", padding="same")(conc2_traj)
        up_conv2_traj = BatchNormalization()(up_conv2_traj)
        up_conv2_traj = Conv2D(128, (3, 3), activation="relu", padding="same")(up_conv2_traj)
        up_conv2_traj = BatchNormalization()(up_conv2_traj)

        conv_t1_traj = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(up_conv2_traj)
        conc1_traj = concatenate([conv_t1_traj, conv1_traj])
        up_conv1_traj = Conv2D(64, (3, 3), activation="relu", padding="same")(conc1_traj)
        up_conv1_traj = BatchNormalization()(up_conv1_traj)
        up_conv1_traj = Conv2D(64, (3, 3), activation="relu", padding="same")(up_conv1_traj)
        up_conv1_traj = BatchNormalization()(up_conv1_traj)

        output_layer1 = Conv2D(2, (1, 1), padding="same", activation="sigmoid")(up_conv1_sat)
        output_layer2 = Conv2D(2, (1, 1), padding="same", activation="sigmoid")(up_conv1_traj)

        fusion_layer = get_fusion_layer(fusion_type, output_layer1, output_layer2)

        if fusion_type == "concat":
            fusion_layer = Conv2D(2, (1, 1), padding="same", activation="sigmoid")(fusion_layer)

        self.model = tf.keras.Model(inputs=inputs_layer, outputs=fusion_layer)

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_function,
                           metrics=get_metrics(batch_size=self.batch_size))

    def resunet_traj_type1(self, dim, input_channels, batch_size, fusion_type):

        """

        ResUnet trajectory late fusion implementation with ResUnet stream for satellite and direct connection to
        trajectory. CAUTION: This model works only with satellite and trajectory data and automatically pick last
        array as trajectory array.

        :param dim: dimension of inputs
        :param input_channels: number of bands/layers of input
        :param batch_size: # batches in the input
        :param fusion_type: preferred fusion type to be used
        :return: compiled model
        """

        inputs_layer = tf.keras.layers.Input((dim[0], dim[1], input_channels), batch_size=batch_size)

        # split input into satellite and trajectory tensors
        input_sat, input_traj = tf.split(inputs_layer, [input_channels - 1, 1], axis=3)

        # satellite resunet stream
        conv1_sat = Conv2D(64, (3, 3), padding="same", strides=1)(input_sat)
        conv1_sat = BatchNormalization()(conv1_sat)
        conv1_sat = Activation("relu")(conv1_sat)
        conv1_sat = Conv2D(64, (3, 3), padding="same", strides=1)(conv1_sat)
        conv1_sat = BatchNormalization()(conv1_sat)

        res1_sat = Conv2D(64, (1, 1), padding="same", strides=1)(input_sat)
        res1_sat = BatchNormalization()(res1_sat)

        conv1_sat_output = Add()([conv1_sat, res1_sat])
        conv1_sat_output = Activation("relu")(conv1_sat_output)

        conv2_sat = Conv2D(128, (3, 3), padding="same", strides=2)(conv1_sat_output)
        conv2_sat = BatchNormalization()(conv2_sat)
        conv2_sat = Activation("relu")(conv2_sat)
        conv2_sat = Conv2D(128, (3, 3), padding="same", strides=1)(conv2_sat)
        conv2_sat = BatchNormalization()(conv2_sat)

        res2_sat = Conv2D(128, (1, 1), padding="same", strides=2)(conv1_sat_output)
        res2_sat = BatchNormalization()(res2_sat)

        conv2_sat_output = Add()([conv2_sat, res2_sat])
        conv2_sat_output = Activation("relu")(conv2_sat_output)

        conv3_sat = Conv2D(256, (3, 3), padding="same", strides=2)(conv2_sat_output)
        conv3_sat = BatchNormalization()(conv3_sat)
        conv3_sat = Activation("relu")(conv3_sat)
        conv3_sat = Conv2D(256, (3, 3), padding="same", strides=1)(conv3_sat)
        conv3_sat = BatchNormalization()(conv3_sat)

        res3_sat = Conv2D(256, (1, 1), padding="same", strides=2)(conv2_sat_output)
        res3_sat = BatchNormalization()(res3_sat)

        conv3_sat_output = Add()([conv3_sat, res3_sat])
        conv3_sat_output = Activation("relu")(conv3_sat_output)

        conv4_sat = Conv2D(512, (3, 3), padding="same", strides=2)(conv3_sat_output)
        conv4_sat = BatchNormalization()(conv4_sat)
        conv4_sat = Activation("relu")(conv4_sat)
        conv4_sat = Conv2D(512, (3, 3), padding="same", strides=1)(conv4_sat)
        conv4_sat = BatchNormalization()(conv4_sat)

        res4_sat = Conv2D(512, (1, 1), padding="same", strides=2)(conv3_sat_output)
        res4_sat = BatchNormalization()(res4_sat)

        conv4_sat_output = Add()([conv4_sat, res4_sat])
        conv4_sat_output = Activation("relu")(conv4_sat_output)

        conv_middle_sat = Conv2D(1024, (3, 3), padding="same", strides=2)(conv4_sat_output)
        conv_middle_sat = BatchNormalization()(conv_middle_sat)
        conv_middle_sat = Activation("relu")(conv_middle_sat)
        conv_middle_sat = Conv2D(1024, (3, 3), padding="same", strides=1)(conv_middle_sat)
        conv_middle_sat = BatchNormalization()(conv_middle_sat)

        res_middle_sat = Conv2D(1024, (1, 1), padding="same", strides=2)(conv4_sat_output)
        res_middle_sat = BatchNormalization()(res_middle_sat)

        conv_middle_sat_output = Add()([conv_middle_sat, res_middle_sat])
        conv_middle_sat_output = Activation("relu")(conv_middle_sat_output)

        upconv1_sat_trans = Conv2DTranspose(512, (3, 3), padding="same", strides=2)(conv_middle_sat_output)
        conc1_sat = concatenate([upconv1_sat_trans, conv4_sat_output])

        upconv1_sat = Conv2D(512, (3, 3), padding="same", strides=1)(conc1_sat)
        upconv1_sat = BatchNormalization()(upconv1_sat)
        upconv1_sat = Activation("relu")(upconv1_sat)
        upconv1_sat = Conv2D(512, (3, 3), padding="same", strides=1)(upconv1_sat)
        upconv1_sat = BatchNormalization()(upconv1_sat)

        res_up_1_sat = Conv2D(512, (1, 1), padding="same", strides=1)(conc1_sat)
        res_up_1_sat = BatchNormalization()(res_up_1_sat)

        upconv1_sat_output = Add()([upconv1_sat, res_up_1_sat])
        upconv1_sat_output = Activation("relu")(upconv1_sat_output)

        upconv2_sat_trans = Conv2DTranspose(256, (3, 3), padding="same", strides=2)(upconv1_sat_output)
        conc2_sat = concatenate([upconv2_sat_trans, conv3_sat_output])

        upconv2_sat = Conv2D(256, (3, 3), padding="same", strides=1)(conc2_sat)
        upconv2_sat = BatchNormalization()(upconv2_sat)
        upconv2_sat = Activation("relu")(upconv2_sat)
        upconv2_sat = Conv2D(256, (3, 3), padding="same", strides=1)(upconv2_sat)
        upconv2_sat = BatchNormalization()(upconv2_sat)

        res_up_2_sat = Conv2D(256, (1, 1), padding="same", strides=1)(conc2_sat)
        res_up_2_sat = BatchNormalization()(res_up_2_sat)

        upconv2_sat_output = Add()([upconv2_sat, res_up_2_sat])
        upconv2_sat_output = Activation("relu")(upconv2_sat_output)

        upconv3_sat_trans = Conv2DTranspose(128, (3, 3), padding="same", strides=2)(upconv2_sat_output)
        conc3_sat = concatenate([upconv3_sat_trans, conv2_sat_output])

        upconv3_sat = Conv2D(128, (3, 3), padding="same", strides=1)(conc3_sat)
        upconv3_sat = BatchNormalization()(upconv3_sat)
        upconv3_sat = Activation("relu")(upconv3_sat)
        upconv3_sat = Conv2D(128, (3, 3), padding="same", strides=1)(upconv3_sat)
        upconv3_sat = BatchNormalization()(upconv3_sat)

        res_up_3_sat = Conv2D(128, (1, 1), padding="same", strides=1)(conc3_sat)
        res_up_3_sat = BatchNormalization()(res_up_3_sat)

        upconv3_sat_output = Add()([upconv3_sat, res_up_3_sat])
        upconv3_sat_output = Activation("relu")(upconv3_sat_output)

        upconv4_sat_trans = Conv2DTranspose(64, (3, 3), padding="same", strides=2)(upconv3_sat_output)
        conc4_sat = concatenate([upconv4_sat_trans, conv1_sat_output])

        upconv4_sat = Conv2D(64, (3, 3), padding="same", strides=1)(conc4_sat)
        upconv4_sat = BatchNormalization()(upconv4_sat)
        upconv4_sat = Activation("relu")(upconv4_sat)
        upconv4_sat = Conv2D(64, (3, 3), padding="same", strides=1)(upconv4_sat)
        upconv4_sat = BatchNormalization()(upconv4_sat)

        res_up_4_sat = Conv2D(64, (1, 1), padding="same", strides=1)(conc4_sat)
        res_up_4_sat = BatchNormalization()(res_up_4_sat)

        upconv4_sat_output = Add()([upconv4_sat, res_up_4_sat])
        upconv4_sat_output = Activation("relu")(upconv4_sat_output)

        output_layer1 = Conv2D(2, (1, 1), padding="same", activation="sigmoid")(upconv4_sat_output)
        output_layer2 = Conv2D(2, (1, 1), padding="same", activation="sigmoid")(input_traj)

        fusion_layer = get_fusion_layer(fusion_type, output_layer1, output_layer2)

        if fusion_type == "concat":
            fusion_layer = Conv2D(2, (1, 1), padding="same", activation="sigmoid")(fusion_layer)

        self.model = tf.keras.Model(inputs=inputs_layer, outputs=fusion_layer)

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_function,
                           metrics=get_metrics(batch_size=self.batch_size))

    def resunet_traj_type2(self, dim, input_channels, batch_size, fusion_type):

        """

        ResUnet trajectory late fusion implementation with ResUnet stream for both satellite and
        trajectory. CAUTION: This model works only with satellite and trajectory data and automatically pick last
        array as trajectory array.

        :param dim: dimension of inputs
        :param input_channels: number of bands/layers of input
        :param batch_size: # batches in the input
        :param fusion_type: preferred fusion type to be used
        :return: compiled model
        """

        inputs_layer = tf.keras.layers.Input((dim[0], dim[1], input_channels), batch_size=batch_size)

        # split input into satellite and trajectory tensors
        input_sat, input_traj = tf.split(inputs_layer, [input_channels - 1, 1], axis=3)

        # satellite resunet stream
        conv1_sat = Conv2D(64, (3, 3), padding="same", strides=1)(input_sat)
        conv1_sat = BatchNormalization()(conv1_sat)
        conv1_sat = Activation("relu")(conv1_sat)
        conv1_sat = Conv2D(64, (3, 3), padding="same", strides=1)(conv1_sat)
        conv1_sat = BatchNormalization()(conv1_sat)

        res1_sat = Conv2D(64, (1, 1), padding="same", strides=1)(input_sat)
        res1_sat = BatchNormalization()(res1_sat)

        conv1_sat_output = Add()([conv1_sat, res1_sat])
        conv1_sat_output = Activation("relu")(conv1_sat_output)

        conv2_sat = Conv2D(128, (3, 3), padding="same", strides=2)(conv1_sat_output)
        conv2_sat = BatchNormalization()(conv2_sat)
        conv2_sat = Activation("relu")(conv2_sat)
        conv2_sat = Conv2D(128, (3, 3), padding="same", strides=1)(conv2_sat)
        conv2_sat = BatchNormalization()(conv2_sat)

        res2_sat = Conv2D(128, (1, 1), padding="same", strides=2)(conv1_sat_output)
        res2_sat = BatchNormalization()(res2_sat)

        conv2_sat_output = Add()([conv2_sat, res2_sat])
        conv2_sat_output = Activation("relu")(conv2_sat_output)

        conv3_sat = Conv2D(256, (3, 3), padding="same", strides=2)(conv2_sat_output)
        conv3_sat = BatchNormalization()(conv3_sat)
        conv3_sat = Activation("relu")(conv3_sat)
        conv3_sat = Conv2D(256, (3, 3), padding="same", strides=1)(conv3_sat)
        conv3_sat = BatchNormalization()(conv3_sat)

        res3_sat = Conv2D(256, (1, 1), padding="same", strides=2)(conv2_sat_output)
        res3_sat = BatchNormalization()(res3_sat)

        conv3_sat_output = Add()([conv3_sat, res3_sat])
        conv3_sat_output = Activation("relu")(conv3_sat_output)

        conv4_sat = Conv2D(512, (3, 3), padding="same", strides=2)(conv3_sat_output)
        conv4_sat = BatchNormalization()(conv4_sat)
        conv4_sat = Activation("relu")(conv4_sat)
        conv4_sat = Conv2D(512, (3, 3), padding="same", strides=1)(conv4_sat)
        conv4_sat = BatchNormalization()(conv4_sat)

        res4_sat = Conv2D(512, (1, 1), padding="same", strides=2)(conv3_sat_output)
        res4_sat = BatchNormalization()(res4_sat)

        conv4_sat_output = Add()([conv4_sat, res4_sat])
        conv4_sat_output = Activation("relu")(conv4_sat_output)

        conv_middle_sat = Conv2D(1024, (3, 3), padding="same", strides=2)(conv4_sat_output)
        conv_middle_sat = BatchNormalization()(conv_middle_sat)
        conv_middle_sat = Activation("relu")(conv_middle_sat)
        conv_middle_sat = Conv2D(1024, (3, 3), padding="same", strides=1)(conv_middle_sat)
        conv_middle_sat = BatchNormalization()(conv_middle_sat)

        res_middle_sat = Conv2D(1024, (1, 1), padding="same", strides=2)(conv4_sat_output)
        res_middle_sat = BatchNormalization()(res_middle_sat)

        conv_middle_sat_output = Add()([conv_middle_sat, res_middle_sat])
        conv_middle_sat_output = Activation("relu")(conv_middle_sat_output)

        upconv1_sat_trans = Conv2DTranspose(512, (3, 3), padding="same", strides=2)(conv_middle_sat_output)
        conc1_sat = concatenate([upconv1_sat_trans, conv4_sat_output])

        upconv1_sat = Conv2D(512, (3, 3), padding="same", strides=1)(conc1_sat)
        upconv1_sat = BatchNormalization()(upconv1_sat)
        upconv1_sat = Activation("relu")(upconv1_sat)
        upconv1_sat = Conv2D(512, (3, 3), padding="same", strides=1)(upconv1_sat)
        upconv1_sat = BatchNormalization()(upconv1_sat)

        res_up_1_sat = Conv2D(512, (1, 1), padding="same", strides=1)(conc1_sat)
        res_up_1_sat = BatchNormalization()(res_up_1_sat)

        upconv1_sat_output = Add()([upconv1_sat, res_up_1_sat])
        upconv1_sat_output = Activation("relu")(upconv1_sat_output)

        upconv2_sat_trans = Conv2DTranspose(256, (3, 3), padding="same", strides=2)(upconv1_sat_output)
        conc2_sat = concatenate([upconv2_sat_trans, conv3_sat_output])

        upconv2_sat = Conv2D(256, (3, 3), padding="same", strides=1)(conc2_sat)
        upconv2_sat = BatchNormalization()(upconv2_sat)
        upconv2_sat = Activation("relu")(upconv2_sat)
        upconv2_sat = Conv2D(256, (3, 3), padding="same", strides=1)(upconv2_sat)
        upconv2_sat = BatchNormalization()(upconv2_sat)

        res_up_2_sat = Conv2D(256, (1, 1), padding="same", strides=1)(conc2_sat)
        res_up_2_sat = BatchNormalization()(res_up_2_sat)

        upconv2_sat_output = Add()([upconv2_sat, res_up_2_sat])
        upconv2_sat_output = Activation("relu")(upconv2_sat_output)

        upconv3_sat_trans = Conv2DTranspose(128, (3, 3), padding="same", strides=2)(upconv2_sat_output)
        conc3_sat = concatenate([upconv3_sat_trans, conv2_sat_output])

        upconv3_sat = Conv2D(128, (3, 3), padding="same", strides=1)(conc3_sat)
        upconv3_sat = BatchNormalization()(upconv3_sat)
        upconv3_sat = Activation("relu")(upconv3_sat)
        upconv3_sat = Conv2D(128, (3, 3), padding="same", strides=1)(upconv3_sat)
        upconv3_sat = BatchNormalization()(upconv3_sat)

        res_up_3_sat = Conv2D(128, (1, 1), padding="same", strides=1)(conc3_sat)
        res_up_3_sat = BatchNormalization()(res_up_3_sat)

        upconv3_sat_output = Add()([upconv3_sat, res_up_3_sat])
        upconv3_sat_output = Activation("relu")(upconv3_sat_output)

        upconv4_sat_trans = Conv2DTranspose(64, (3, 3), padding="same", strides=2)(upconv3_sat_output)
        conc4_sat = concatenate([upconv4_sat_trans, conv1_sat_output])

        upconv4_sat = Conv2D(64, (3, 3), padding="same", strides=1)(conc4_sat)
        upconv4_sat = BatchNormalization()(upconv4_sat)
        upconv4_sat = Activation("relu")(upconv4_sat)
        upconv4_sat = Conv2D(64, (3, 3), padding="same", strides=1)(upconv4_sat)
        upconv4_sat = BatchNormalization()(upconv4_sat)

        res_up_4_sat = Conv2D(64, (1, 1), padding="same", strides=1)(conc4_sat)
        res_up_4_sat = BatchNormalization()(res_up_4_sat)

        upconv4_sat_output = Add()([upconv4_sat, res_up_4_sat])
        upconv4_sat_output = Activation("relu")(upconv4_sat_output)

        # trajectory resunet stream
        conv1_traj = Conv2D(64, (3, 3), padding="same", strides=1)(input_traj)
        conv1_traj = BatchNormalization()(conv1_traj)
        conv1_traj = Activation("relu")(conv1_traj)
        conv1_traj = Conv2D(64, (3, 3), padding="same", strides=1)(conv1_traj)
        conv1_traj = BatchNormalization()(conv1_traj)

        res1_traj = Conv2D(64, (1, 1), padding="same", strides=1)(input_traj)
        res1_traj = BatchNormalization()(res1_traj)

        conv1_traj_output = Add()([conv1_traj, res1_traj])
        conv1_traj_output = Activation("relu")(conv1_traj_output)

        conv2_traj = Conv2D(128, (3, 3), padding="same", strides=2)(conv1_traj_output)
        conv2_traj = BatchNormalization()(conv2_traj)
        conv2_traj = Activation("relu")(conv2_traj)
        conv2_traj = Conv2D(128, (3, 3), padding="same", strides=1)(conv2_traj)
        conv2_traj = BatchNormalization()(conv2_traj)

        res2_traj = Conv2D(128, (1, 1), padding="same", strides=2)(conv1_traj_output)
        res2_traj = BatchNormalization()(res2_traj)

        conv2_traj_output = Add()([conv2_traj, res2_traj])
        conv2_traj_output = Activation("relu")(conv2_traj_output)

        conv3_traj = Conv2D(256, (3, 3), padding="same", strides=2)(conv2_traj_output)
        conv3_traj = BatchNormalization()(conv3_traj)
        conv3_traj = Activation("relu")(conv3_traj)
        conv3_traj = Conv2D(256, (3, 3), padding="same", strides=1)(conv3_traj)
        conv3_traj = BatchNormalization()(conv3_traj)

        res3_traj = Conv2D(256, (1, 1), padding="same", strides=2)(conv2_traj_output)
        res3_traj = BatchNormalization()(res3_traj)

        conv3_traj_output = Add()([conv3_traj, res3_traj])
        conv3_traj_output = Activation("relu")(conv3_traj_output)

        conv4_traj = Conv2D(512, (3, 3), padding="same", strides=2)(conv3_traj_output)
        conv4_traj = BatchNormalization()(conv4_traj)
        conv4_traj = Activation("relu")(conv4_traj)
        conv4_traj = Conv2D(512, (3, 3), padding="same", strides=1)(conv4_traj)
        conv4_traj = BatchNormalization()(conv4_traj)

        res4_traj = Conv2D(512, (1, 1), padding="same", strides=2)(conv3_traj_output)
        res4_traj = BatchNormalization()(res4_traj)

        conv4_traj_output = Add()([conv4_traj, res4_traj])
        conv4_traj_output = Activation("relu")(conv4_traj_output)

        conv_middle_traj = Conv2D(1024, (3, 3), padding="same", strides=2)(conv4_traj_output)
        conv_middle_traj = BatchNormalization()(conv_middle_traj)
        conv_middle_traj = Activation("relu")(conv_middle_traj)
        conv_middle_traj = Conv2D(1024, (3, 3), padding="same", strides=1)(conv_middle_traj)
        conv_middle_traj = BatchNormalization()(conv_middle_traj)

        res_middle_traj = Conv2D(1024, (1, 1), padding="same", strides=2)(conv4_traj_output)
        res_middle_traj = BatchNormalization()(res_middle_traj)

        conv_middle_traj_output = Add()([conv_middle_traj, res_middle_traj])
        conv_middle_traj_output = Activation("relu")(conv_middle_traj_output)

        upconv1_traj_trans = Conv2DTranspose(512, (3, 3), padding="same", strides=2)(conv_middle_traj_output)
        conc1_traj = concatenate([upconv1_traj_trans, conv4_traj_output])

        upconv1_traj = Conv2D(512, (3, 3), padding="same", strides=1)(conc1_traj)
        upconv1_traj = BatchNormalization()(upconv1_traj)
        upconv1_traj = Activation("relu")(upconv1_traj)
        upconv1_traj = Conv2D(512, (3, 3), padding="same", strides=1)(upconv1_traj)
        upconv1_traj = BatchNormalization()(upconv1_traj)

        res_up_1_traj = Conv2D(512, (1, 1), padding="same", strides=1)(conc1_traj)
        res_up_1_traj = BatchNormalization()(res_up_1_traj)

        upconv1_traj_output = Add()([upconv1_traj, res_up_1_traj])
        upconv1_traj_output = Activation("relu")(upconv1_traj_output)

        upconv2_traj_trans = Conv2DTranspose(256, (3, 3), padding="same", strides=2)(upconv1_traj_output)
        conc2_traj = concatenate([upconv2_traj_trans, conv3_traj_output])

        upconv2_traj = Conv2D(256, (3, 3), padding="same", strides=1)(conc2_traj)
        upconv2_traj = BatchNormalization()(upconv2_traj)
        upconv2_traj = Activation("relu")(upconv2_traj)
        upconv2_traj = Conv2D(256, (3, 3), padding="same", strides=1)(upconv2_traj)
        upconv2_traj = BatchNormalization()(upconv2_traj)

        res_up_2_traj = Conv2D(256, (1, 1), padding="same", strides=1)(conc2_traj)
        res_up_2_traj = BatchNormalization()(res_up_2_traj)

        upconv2_traj_output = Add()([upconv2_traj, res_up_2_traj])
        upconv2_traj_output = Activation("relu")(upconv2_traj_output)

        upconv3_traj_trans = Conv2DTranspose(128, (3, 3), padding="same", strides=2)(upconv2_traj_output)
        conc3_traj = concatenate([upconv3_traj_trans, conv2_traj_output])

        upconv3_traj = Conv2D(128, (3, 3), padding="same", strides=1)(conc3_traj)
        upconv3_traj = BatchNormalization()(upconv3_traj)
        upconv3_traj = Activation("relu")(upconv3_traj)
        upconv3_traj = Conv2D(128, (3, 3), padding="same", strides=1)(upconv3_traj)
        upconv3_traj = BatchNormalization()(upconv3_traj)

        res_up_3_traj = Conv2D(128, (1, 1), padding="same", strides=1)(conc3_traj)
        res_up_3_traj = BatchNormalization()(res_up_3_traj)

        upconv3_traj_output = Add()([upconv3_traj, res_up_3_traj])
        upconv3_traj_output = Activation("relu")(upconv3_traj_output)

        upconv4_traj_trans = Conv2DTranspose(64, (3, 3), padding="same", strides=2)(upconv3_traj_output)
        conc4_traj = concatenate([upconv4_traj_trans, conv1_traj_output])

        upconv4_traj = Conv2D(64, (3, 3), padding="same", strides=1)(conc4_traj)
        upconv4_traj = BatchNormalization()(upconv4_traj)
        upconv4_traj = Activation("relu")(upconv4_traj)
        upconv4_traj = Conv2D(64, (3, 3), padding="same", strides=1)(upconv4_traj)
        upconv4_traj = BatchNormalization()(upconv4_traj)

        res_up_4_traj = Conv2D(64, (1, 1), padding="same", strides=1)(conc4_traj)
        res_up_4_traj = BatchNormalization()(res_up_4_traj)

        upconv4_traj_output = Add()([upconv4_traj, res_up_4_traj])
        upconv4_traj_output = Activation("relu")(upconv4_traj_output)

        output_layer1 = Conv2D(2, (1, 1), padding="same", activation="sigmoid")(upconv4_sat_output)
        output_layer2 = Conv2D(2, (1, 1), padding="same", activation="sigmoid")(upconv4_traj_output)

        fusion_layer = get_fusion_layer(fusion_type, output_layer1, output_layer2)

        if fusion_type == "concat":
            fusion_layer = Conv2D(2, (1, 1), padding="same", activation="sigmoid")(fusion_layer)

        self.model = tf.keras.Model(inputs=inputs_layer, outputs=fusion_layer)

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_function,
                           metrics=get_metrics(batch_size=self.batch_size))

    def dlinknet_traj_type1(self, dim, input_channels, batch_size, fusion_type):

        """
        D-Linknet trajectory late fusion implementation with D-Linknet stream for satellite and direct connection to
        trajectory. CAUTION: This model works only with satellite and trajectory data and automatically pick last
        array as trajectory array.

        :param dim: dimension of inputs
        :param input_channels: number of bands/layers of input
        :param batch_size: # batches in the input
        :param fusion_type: preferred fusion type to be used
        :return:
        """

        inputs_layer = tf.keras.layers.Input((dim[0], dim[1], input_channels), batch_size=batch_size)

        # split input into satellite and trajectory tensors
        input_sat, input_traj = tf.split(inputs_layer, [input_channels - 1, 1], axis=3)

        # satellite d-linknet stream
        conv1_sat = Conv2D(64, (7, 7), padding="same", strides=2)(input_sat)
        conv1_sat = BatchNormalization()(conv1_sat)
        conv1_sat = Activation("relu")(conv1_sat)
        conv1_sat = MaxPooling2D((3, 3), strides=2, padding="same")(conv1_sat)

        # satellite level1
        conv2_sat = Conv2D(64, (3, 3), padding="same", strides=1)(conv1_sat)
        conv2_sat = BatchNormalization()(conv2_sat)
        conv2_sat = Activation("relu")(conv2_sat)
        conv2_sat = Conv2D(64, (3, 3), padding="same", strides=1)(conv2_sat)
        conv2_sat = BatchNormalization()(conv2_sat)
        conv2_sat = Activation("relu")(conv2_sat)
        conv2_sat_add = Add()([conv2_sat, conv1_sat])
        conv2_sat_add = BatchNormalization()(conv2_sat_add)
        conv2_sat_add = Activation("relu")(conv2_sat_add)

        conv3_sat = Conv2D(64, (3, 3), padding="same", strides=1)(conv2_sat_add)
        conv3_sat = BatchNormalization()(conv3_sat)
        conv3_sat = Activation("relu")(conv3_sat)
        conv3_sat = Conv2D(64, (3, 3), padding="same", strides=1)(conv3_sat)
        conv3_sat = BatchNormalization()(conv3_sat)
        conv3_sat = Activation("relu")(conv3_sat)
        conv3_sat_add = Add()([conv3_sat, conv2_sat_add])
        conv3_sat_add = BatchNormalization()(conv3_sat_add)
        conv3_sat_add = Activation("relu")(conv3_sat_add)

        conv4_sat = Conv2D(64, (3, 3), padding="same", strides=1)(conv3_sat_add)
        conv4_sat = BatchNormalization()(conv4_sat)
        conv4_sat = Activation("relu")(conv4_sat)
        conv4_sat = Conv2D(64, (3, 3), padding="same", strides=1)(conv4_sat)
        conv4_sat = BatchNormalization()(conv4_sat)
        conv4_sat = Activation("relu")(conv4_sat)
        conv4_sat_add = Add()([conv4_sat, conv3_sat_add])
        conv4_sat_add = BatchNormalization()(conv4_sat_add)
        conv4_sat_add = Activation("relu")(conv4_sat_add)

        # satellite level2
        conv5_sat = Conv2D(128, (3, 3), padding="same", strides=2)(conv4_sat_add)
        conv5_sat = BatchNormalization()(conv5_sat)
        conv5_sat = Activation("relu")(conv5_sat)
        conv5_sat = Conv2D(128, (3, 3), padding="same", strides=1)(conv5_sat)
        conv5_sat = BatchNormalization()(conv5_sat)
        conv5_sat = Activation("relu")(conv5_sat)
        conv4_sat_add_skip = Conv2D(128, (1, 1), padding="same", strides=2)(conv4_sat_add)
        conv5_sat_add = Add()([conv5_sat, conv4_sat_add_skip])
        conv5_sat_add = BatchNormalization()(conv5_sat_add)
        conv5_sat_add = Activation("relu")(conv5_sat_add)

        conv6_sat = Conv2D(128, (3, 3), padding="same", strides=1)(conv5_sat_add)
        conv6_sat = BatchNormalization()(conv6_sat)
        conv6_sat = Activation("relu")(conv6_sat)
        conv6_sat = Conv2D(128, (3, 3), padding="same", strides=1)(conv6_sat)
        conv6_sat = BatchNormalization()(conv6_sat)
        conv6_sat = Activation("relu")(conv6_sat)
        conv6_sat_add = Add()([conv6_sat, conv5_sat_add])
        conv6_sat_add = BatchNormalization()(conv6_sat_add)
        conv6_sat_add = Activation("relu")(conv6_sat_add)

        conv7_sat = Conv2D(128, (3, 3), padding="same", strides=1)(conv6_sat_add)
        conv7_sat = BatchNormalization()(conv7_sat)
        conv7_sat = Activation("relu")(conv7_sat)
        conv7_sat = Conv2D(128, (3, 3), padding="same", strides=1)(conv7_sat)
        conv7_sat = BatchNormalization()(conv7_sat)
        conv7_sat = Activation("relu")(conv7_sat)
        conv7_sat_add = Add()([conv7_sat, conv6_sat_add])
        conv7_sat_add = BatchNormalization()(conv7_sat_add)
        conv7_sat_add = Activation("relu")(conv7_sat_add)

        conv8_sat = Conv2D(128, (3, 3), padding="same", strides=1)(conv7_sat_add)
        conv8_sat = BatchNormalization()(conv8_sat)
        conv8_sat = Activation("relu")(conv8_sat)
        conv8_sat = Conv2D(128, (3, 3), padding="same", strides=1)(conv8_sat)
        conv8_sat = BatchNormalization()(conv8_sat)
        conv8_sat = Activation("relu")(conv8_sat)
        conv8_sat_add = Add()([conv8_sat, conv7_sat_add])
        conv8_sat_add = BatchNormalization()(conv8_sat_add)
        conv8_sat_add = Activation("relu")(conv8_sat_add)

        # satellite level3
        conv9_sat = Conv2D(256, (3, 3), padding="same", strides=2)(conv8_sat_add)
        conv9_sat = BatchNormalization()(conv9_sat)
        conv9_sat = Activation("relu")(conv9_sat)
        conv9_sat = Conv2D(256, (3, 3), padding="same", strides=1)(conv9_sat)
        conv9_sat = BatchNormalization()(conv9_sat)
        conv9_sat = Activation("relu")(conv9_sat)
        conv8_sat_add_skip = Conv2D(256, (1, 1), padding="same", strides=2)(conv8_sat_add)
        conv9_sat_add = Add()([conv9_sat, conv8_sat_add_skip])
        conv9_sat_add = BatchNormalization()(conv9_sat_add)
        conv9_sat_add = Activation("relu")(conv9_sat_add)

        conv10_sat = Conv2D(256, (3, 3), padding="same", strides=1)(conv9_sat_add)
        conv10_sat = BatchNormalization()(conv10_sat)
        conv10_sat = Activation("relu")(conv10_sat)
        conv10_sat = Conv2D(256, (3, 3), padding="same", strides=1)(conv10_sat)
        conv10_sat = BatchNormalization()(conv10_sat)
        conv10_sat = Activation("relu")(conv10_sat)
        conv10_sat_add = Add()([conv10_sat, conv9_sat_add])
        conv10_sat_add = BatchNormalization()(conv10_sat_add)
        conv10_sat_add = Activation("relu")(conv10_sat_add)

        conv11_sat = Conv2D(256, (3, 3), padding="same", strides=1)(conv10_sat_add)
        conv11_sat = BatchNormalization()(conv11_sat)
        conv11_sat = Activation("relu")(conv11_sat)
        conv11_sat = Conv2D(256, (3, 3), padding="same", strides=1)(conv11_sat)
        conv11_sat = BatchNormalization()(conv11_sat)
        conv11_sat = Activation("relu")(conv11_sat)
        conv11_sat_add = Add()([conv11_sat, conv10_sat_add])
        conv11_sat_add = BatchNormalization()(conv11_sat_add)
        conv11_sat_add = Activation("relu")(conv11_sat_add)

        conv12_sat = Conv2D(256, (3, 3), padding="same", strides=1)(conv11_sat_add)
        conv12_sat = BatchNormalization()(conv12_sat)
        conv12_sat = Activation("relu")(conv12_sat)
        conv12_sat = Conv2D(256, (3, 3), padding="same", strides=1)(conv12_sat)
        conv12_sat = BatchNormalization()(conv12_sat)
        conv12_sat = Activation("relu")(conv12_sat)
        conv12_sat_add = Add()([conv12_sat, conv11_sat_add])
        conv12_sat_add = BatchNormalization()(conv12_sat_add)
        conv12_sat_add = Activation("relu")(conv12_sat_add)

        conv13_sat = Conv2D(256, (3, 3), padding="same", strides=1)(conv12_sat_add)
        conv13_sat = BatchNormalization()(conv13_sat)
        conv13_sat = Activation("relu")(conv13_sat)
        conv13_sat = Conv2D(256, (3, 3), padding="same", strides=1)(conv13_sat)
        conv13_sat = BatchNormalization()(conv13_sat)
        conv13_sat = Activation("relu")(conv13_sat)
        conv13_sat_add = Add()([conv13_sat, conv12_sat_add])
        conv13_sat_add = BatchNormalization()(conv13_sat_add)
        conv13_sat_add = Activation("relu")(conv13_sat_add)

        conv14_sat = Conv2D(256, (3, 3), padding="same", strides=1)(conv13_sat_add)
        conv14_sat = BatchNormalization()(conv14_sat)
        conv14_sat = Activation("relu")(conv14_sat)
        conv14_sat = Conv2D(256, (3, 3), padding="same", strides=1)(conv14_sat)
        conv14_sat = BatchNormalization()(conv14_sat)
        conv14_sat = Activation("relu")(conv14_sat)
        conv14_sat_add = Add()([conv14_sat, conv13_sat_add])
        conv14_sat_add = BatchNormalization()(conv14_sat_add)
        conv14_sat_add = Activation("relu")(conv14_sat_add)

        # satellite level4
        conv15_sat = Conv2D(512, (3, 3), padding="same", strides=2)(conv14_sat_add)
        conv15_sat = BatchNormalization()(conv15_sat)
        conv15_sat = Activation("relu")(conv15_sat)
        conv15_sat = Conv2D(512, (3, 3), padding="same", strides=1)(conv15_sat)
        conv15_sat = BatchNormalization()(conv15_sat)
        conv15_sat = Activation("relu")(conv15_sat)
        conv14_sat_add_skip = Conv2D(512, (1, 1), padding="same", strides=2)(conv14_sat_add)
        conv15_sat_add = Add()([conv15_sat, conv14_sat_add_skip])
        conv15_sat_add = BatchNormalization()(conv15_sat_add)
        conv15_sat_add = Activation("relu")(conv15_sat_add)

        conv16_sat = Conv2D(512, (3, 3), padding="same", strides=1)(conv15_sat_add)
        conv16_sat = BatchNormalization()(conv16_sat)
        conv16_sat = Activation("relu")(conv16_sat)
        conv16_sat = Conv2D(512, (3, 3), padding="same", strides=1)(conv16_sat)
        conv16_sat = BatchNormalization()(conv16_sat)
        conv16_sat = Activation("relu")(conv16_sat)
        conv16_sat_add = Add()([conv16_sat, conv15_sat_add])
        conv16_sat_add = BatchNormalization()(conv16_sat_add)
        conv16_sat_add = Activation("relu")(conv16_sat_add)

        conv17_sat = Conv2D(512, (3, 3), padding="same", strides=1)(conv16_sat_add)
        conv17_sat = BatchNormalization()(conv17_sat)
        conv17_sat = Activation("relu")(conv17_sat)
        conv17_sat = Conv2D(512, (3, 3), padding="same", strides=1)(conv17_sat)
        conv17_sat = BatchNormalization()(conv17_sat)
        conv17_sat = Activation("relu")(conv17_sat)
        conv17_sat_add = Add()([conv17_sat, conv16_sat_add])
        conv17_sat_add = BatchNormalization()(conv17_sat_add)
        conv17_sat_add = Activation("relu")(conv17_sat_add)

        last_pool_sat = AveragePooling2D((2, 2), padding="same", strides=1)(conv17_sat_add)

        # satellite dilation8
        dilation8_sat = Conv2D(512, (3, 3), dilation_rate=1, padding="same")(last_pool_sat)
        dilation8_sat = Activation("relu")(dilation8_sat)
        dilation8_sat = Conv2D(512, (3, 3), dilation_rate=2, padding="same")(dilation8_sat)
        dilation8_sat = Activation("relu")(dilation8_sat)
        dilation8_sat = Conv2D(512, (3, 3), dilation_rate=4, padding="same")(dilation8_sat)
        dilation8_sat = Activation("relu")(dilation8_sat)
        dilation8_sat = Conv2D(512, (3, 3), dilation_rate=8, padding="same")(dilation8_sat)
        dilation8_sat = Activation("relu")(dilation8_sat)

        # satellite dilation4
        dilation4_sat = Conv2D(512, (3, 3), dilation_rate=1, padding="same")(last_pool_sat)
        dilation4_sat = Activation("relu")(dilation4_sat)
        dilation4_sat = Conv2D(512, (3, 3), dilation_rate=2, padding="same")(dilation4_sat)
        dilation4_sat = Activation("relu")(dilation4_sat)
        dilation4_sat = Conv2D(512, (3, 3), dilation_rate=4, padding="same")(dilation4_sat)
        dilation4_sat = Activation("relu")(dilation4_sat)

        # satellite dilation2
        dilation2_sat = Conv2D(512, (3, 3), dilation_rate=1, padding="same")(last_pool_sat)
        dilation2_sat = Activation("relu")(dilation2_sat)
        dilation2_sat = Conv2D(512, (3, 3), dilation_rate=2, padding="same")(dilation2_sat)
        dilation2_sat = Activation("relu")(dilation2_sat)

        # satellite dilation1
        dilation1_sat = Conv2D(512, (3, 3), dilation_rate=1, padding="same")(last_pool_sat)
        dilation1_sat = Activation("relu")(dilation1_sat)

        dilation_sat_out = Add()([last_pool_sat, dilation1_sat, dilation2_sat, dilation4_sat, dilation8_sat])

        # satellite decode level3
        decode3_sat = Conv2D(512, (1, 1), padding="same", strides=1)(dilation_sat_out)
        decode3_sat = BatchNormalization()(decode3_sat)
        decode3_sat = Activation("relu")(decode3_sat)

        decode3_sat = Conv2DTranspose(128, (3, 3), strides=2, padding="same")(decode3_sat)
        decode3_sat = BatchNormalization()(decode3_sat)
        decode3_sat = Activation("relu")(decode3_sat)

        decode3_sat = Conv2D(256, (1, 1), strides=1, padding="same")(decode3_sat)
        decode3_sat = BatchNormalization()(decode3_sat)
        decode3_sat = Activation("relu")(decode3_sat)

        decode3_sat_out = Add()([decode3_sat, conv14_sat_add])

        # satellite decode level2
        decode2_sat = Conv2D(256, (1, 1), padding="same", strides=1)(decode3_sat_out)
        decode2_sat = BatchNormalization()(decode2_sat)
        decode2_sat = Activation("relu")(decode2_sat)

        decode2_sat = Conv2DTranspose(64, (3, 3), strides=2, padding="same")(decode2_sat)
        decode2_sat = BatchNormalization()(decode2_sat)
        decode2_sat = Activation("relu")(decode2_sat)

        decode2_sat = Conv2D(128, (1, 1), strides=1, padding="same")(decode2_sat)
        decode2_sat = BatchNormalization()(decode2_sat)
        decode2_sat = Activation("relu")(decode2_sat)

        decode2_sat_out = Add()([decode2_sat, conv8_sat_add])

        # satellite decode level1
        decode1_sat = Conv2D(128, (1, 1), padding="same", strides=1)(decode2_sat_out)
        decode1_sat = BatchNormalization()(decode1_sat)
        decode1_sat = Activation("relu")(decode1_sat)

        decode1_sat = Conv2DTranspose(32, (3, 3), strides=2, padding="same")(decode1_sat)
        decode1_sat = BatchNormalization()(decode1_sat)
        decode1_sat = Activation("relu")(decode1_sat)

        decode1_sat = Conv2D(64, (1, 1), strides=1, padding="same")(decode1_sat)
        decode1_sat = BatchNormalization()(decode1_sat)
        decode1_sat = Activation("relu")(decode1_sat)

        # satellite decode out
        decode_sat_out = Add()([decode1_sat, conv4_sat_add])

        final_conv_sat = Conv2DTranspose(64, (3, 3), padding="same", strides=2)(decode_sat_out)
        final_conv_sat = Activation("relu")(final_conv_sat)
        final_conv_sat = Conv2D(32, (3, 3), padding="same")(final_conv_sat)
        final_conv_sat = Activation("relu")(final_conv_sat)

        output_layer1 = Conv2D(2, (1, 1), padding="same", activation="sigmoid")(final_conv_sat)
        output_layer2 = Conv2D(2, (1, 1), padding="same", activation="sigmoid")(input_traj)

        fusion_layer = get_fusion_layer(fusion_type, output_layer1, output_layer2)

        if fusion_type == "concat":
            fusion_layer = Conv2D(2, (1, 1), padding="same", activation="sigmoid")(fusion_layer)

        self.model = tf.keras.Model(inputs=inputs_layer, outputs=fusion_layer)

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_function,
                           metrics=get_metrics(batch_size=self.batch_size))

    def dlinknet_traj_type2(self, dim, input_channels, batch_size, fusion_type):

        """
        D-Linknet trajectory late fusion implementation with D-Linknet stream for both satellite and trajectory.
        CAUTION: This model works only with satellite and trajectory data and automatically pick last array as
        trajectory array.

        :param dim: dimension of inputs
        :param input_channels: number of bands/layers of input
        :param batch_size: # batches in the input
        :param fusion_type: preferred fusion type to be used
        :return:
        """

        inputs_layer = tf.keras.layers.Input((dim[0], dim[1], input_channels), batch_size=batch_size)

        # split input into satellite and trajectory tensors
        input_sat, input_traj = tf.split(inputs_layer, [input_channels - 1, 1], axis=3)

        # satellite d-linknet stream
        conv1_sat = Conv2D(64, (7, 7), padding="same", strides=2)(input_sat)
        conv1_sat = BatchNormalization()(conv1_sat)
        conv1_sat = Activation("relu")(conv1_sat)
        conv1_sat = MaxPooling2D((3, 3), strides=2, padding="same")(conv1_sat)

        # satellite level1
        conv2_sat = Conv2D(64, (3, 3), padding="same", strides=1)(conv1_sat)
        conv2_sat = BatchNormalization()(conv2_sat)
        conv2_sat = Activation("relu")(conv2_sat)
        conv2_sat = Conv2D(64, (3, 3), padding="same", strides=1)(conv2_sat)
        conv2_sat = BatchNormalization()(conv2_sat)
        conv2_sat = Activation("relu")(conv2_sat)
        conv2_sat_add = Add()([conv2_sat, conv1_sat])
        conv2_sat_add = BatchNormalization()(conv2_sat_add)
        conv2_sat_add = Activation("relu")(conv2_sat_add)

        conv3_sat = Conv2D(64, (3, 3), padding="same", strides=1)(conv2_sat_add)
        conv3_sat = BatchNormalization()(conv3_sat)
        conv3_sat = Activation("relu")(conv3_sat)
        conv3_sat = Conv2D(64, (3, 3), padding="same", strides=1)(conv3_sat)
        conv3_sat = BatchNormalization()(conv3_sat)
        conv3_sat = Activation("relu")(conv3_sat)
        conv3_sat_add = Add()([conv3_sat, conv2_sat_add])
        conv3_sat_add = BatchNormalization()(conv3_sat_add)
        conv3_sat_add = Activation("relu")(conv3_sat_add)

        conv4_sat = Conv2D(64, (3, 3), padding="same", strides=1)(conv3_sat_add)
        conv4_sat = BatchNormalization()(conv4_sat)
        conv4_sat = Activation("relu")(conv4_sat)
        conv4_sat = Conv2D(64, (3, 3), padding="same", strides=1)(conv4_sat)
        conv4_sat = BatchNormalization()(conv4_sat)
        conv4_sat = Activation("relu")(conv4_sat)
        conv4_sat_add = Add()([conv4_sat, conv3_sat_add])
        conv4_sat_add = BatchNormalization()(conv4_sat_add)
        conv4_sat_add = Activation("relu")(conv4_sat_add)

        # satellite level2
        conv5_sat = Conv2D(128, (3, 3), padding="same", strides=2)(conv4_sat_add)
        conv5_sat = BatchNormalization()(conv5_sat)
        conv5_sat = Activation("relu")(conv5_sat)
        conv5_sat = Conv2D(128, (3, 3), padding="same", strides=1)(conv5_sat)
        conv5_sat = BatchNormalization()(conv5_sat)
        conv5_sat = Activation("relu")(conv5_sat)
        conv4_sat_add_skip = Conv2D(128, (1, 1), padding="same", strides=2)(conv4_sat_add)
        conv5_sat_add = Add()([conv5_sat, conv4_sat_add_skip])
        conv5_sat_add = BatchNormalization()(conv5_sat_add)
        conv5_sat_add = Activation("relu")(conv5_sat_add)

        conv6_sat = Conv2D(128, (3, 3), padding="same", strides=1)(conv5_sat_add)
        conv6_sat = BatchNormalization()(conv6_sat)
        conv6_sat = Activation("relu")(conv6_sat)
        conv6_sat = Conv2D(128, (3, 3), padding="same", strides=1)(conv6_sat)
        conv6_sat = BatchNormalization()(conv6_sat)
        conv6_sat = Activation("relu")(conv6_sat)
        conv6_sat_add = Add()([conv6_sat, conv5_sat_add])
        conv6_sat_add = BatchNormalization()(conv6_sat_add)
        conv6_sat_add = Activation("relu")(conv6_sat_add)

        conv7_sat = Conv2D(128, (3, 3), padding="same", strides=1)(conv6_sat_add)
        conv7_sat = BatchNormalization()(conv7_sat)
        conv7_sat = Activation("relu")(conv7_sat)
        conv7_sat = Conv2D(128, (3, 3), padding="same", strides=1)(conv7_sat)
        conv7_sat = BatchNormalization()(conv7_sat)
        conv7_sat = Activation("relu")(conv7_sat)
        conv7_sat_add = Add()([conv7_sat, conv6_sat_add])
        conv7_sat_add = BatchNormalization()(conv7_sat_add)
        conv7_sat_add = Activation("relu")(conv7_sat_add)

        conv8_sat = Conv2D(128, (3, 3), padding="same", strides=1)(conv7_sat_add)
        conv8_sat = BatchNormalization()(conv8_sat)
        conv8_sat = Activation("relu")(conv8_sat)
        conv8_sat = Conv2D(128, (3, 3), padding="same", strides=1)(conv8_sat)
        conv8_sat = BatchNormalization()(conv8_sat)
        conv8_sat = Activation("relu")(conv8_sat)
        conv8_sat_add = Add()([conv8_sat, conv7_sat_add])
        conv8_sat_add = BatchNormalization()(conv8_sat_add)
        conv8_sat_add = Activation("relu")(conv8_sat_add)

        # satellite level3
        conv9_sat = Conv2D(256, (3, 3), padding="same", strides=2)(conv8_sat_add)
        conv9_sat = BatchNormalization()(conv9_sat)
        conv9_sat = Activation("relu")(conv9_sat)
        conv9_sat = Conv2D(256, (3, 3), padding="same", strides=1)(conv9_sat)
        conv9_sat = BatchNormalization()(conv9_sat)
        conv9_sat = Activation("relu")(conv9_sat)
        conv8_sat_add_skip = Conv2D(256, (1, 1), padding="same", strides=2)(conv8_sat_add)
        conv9_sat_add = Add()([conv9_sat, conv8_sat_add_skip])
        conv9_sat_add = BatchNormalization()(conv9_sat_add)
        conv9_sat_add = Activation("relu")(conv9_sat_add)

        conv10_sat = Conv2D(256, (3, 3), padding="same", strides=1)(conv9_sat_add)
        conv10_sat = BatchNormalization()(conv10_sat)
        conv10_sat = Activation("relu")(conv10_sat)
        conv10_sat = Conv2D(256, (3, 3), padding="same", strides=1)(conv10_sat)
        conv10_sat = BatchNormalization()(conv10_sat)
        conv10_sat = Activation("relu")(conv10_sat)
        conv10_sat_add = Add()([conv10_sat, conv9_sat_add])
        conv10_sat_add = BatchNormalization()(conv10_sat_add)
        conv10_sat_add = Activation("relu")(conv10_sat_add)

        conv11_sat = Conv2D(256, (3, 3), padding="same", strides=1)(conv10_sat_add)
        conv11_sat = BatchNormalization()(conv11_sat)
        conv11_sat = Activation("relu")(conv11_sat)
        conv11_sat = Conv2D(256, (3, 3), padding="same", strides=1)(conv11_sat)
        conv11_sat = BatchNormalization()(conv11_sat)
        conv11_sat = Activation("relu")(conv11_sat)
        conv11_sat_add = Add()([conv11_sat, conv10_sat_add])
        conv11_sat_add = BatchNormalization()(conv11_sat_add)
        conv11_sat_add = Activation("relu")(conv11_sat_add)

        conv12_sat = Conv2D(256, (3, 3), padding="same", strides=1)(conv11_sat_add)
        conv12_sat = BatchNormalization()(conv12_sat)
        conv12_sat = Activation("relu")(conv12_sat)
        conv12_sat = Conv2D(256, (3, 3), padding="same", strides=1)(conv12_sat)
        conv12_sat = BatchNormalization()(conv12_sat)
        conv12_sat = Activation("relu")(conv12_sat)
        conv12_sat_add = Add()([conv12_sat, conv11_sat_add])
        conv12_sat_add = BatchNormalization()(conv12_sat_add)
        conv12_sat_add = Activation("relu")(conv12_sat_add)

        conv13_sat = Conv2D(256, (3, 3), padding="same", strides=1)(conv12_sat_add)
        conv13_sat = BatchNormalization()(conv13_sat)
        conv13_sat = Activation("relu")(conv13_sat)
        conv13_sat = Conv2D(256, (3, 3), padding="same", strides=1)(conv13_sat)
        conv13_sat = BatchNormalization()(conv13_sat)
        conv13_sat = Activation("relu")(conv13_sat)
        conv13_sat_add = Add()([conv13_sat, conv12_sat_add])
        conv13_sat_add = BatchNormalization()(conv13_sat_add)
        conv13_sat_add = Activation("relu")(conv13_sat_add)

        conv14_sat = Conv2D(256, (3, 3), padding="same", strides=1)(conv13_sat_add)
        conv14_sat = BatchNormalization()(conv14_sat)
        conv14_sat = Activation("relu")(conv14_sat)
        conv14_sat = Conv2D(256, (3, 3), padding="same", strides=1)(conv14_sat)
        conv14_sat = BatchNormalization()(conv14_sat)
        conv14_sat = Activation("relu")(conv14_sat)
        conv14_sat_add = Add()([conv14_sat, conv13_sat_add])
        conv14_sat_add = BatchNormalization()(conv14_sat_add)
        conv14_sat_add = Activation("relu")(conv14_sat_add)

        # satellite level4
        conv15_sat = Conv2D(512, (3, 3), padding="same", strides=2)(conv14_sat_add)
        conv15_sat = BatchNormalization()(conv15_sat)
        conv15_sat = Activation("relu")(conv15_sat)
        conv15_sat = Conv2D(512, (3, 3), padding="same", strides=1)(conv15_sat)
        conv15_sat = BatchNormalization()(conv15_sat)
        conv15_sat = Activation("relu")(conv15_sat)
        conv14_sat_add_skip = Conv2D(512, (1, 1), padding="same", strides=2)(conv14_sat_add)
        conv15_sat_add = Add()([conv15_sat, conv14_sat_add_skip])
        conv15_sat_add = BatchNormalization()(conv15_sat_add)
        conv15_sat_add = Activation("relu")(conv15_sat_add)

        conv16_sat = Conv2D(512, (3, 3), padding="same", strides=1)(conv15_sat_add)
        conv16_sat = BatchNormalization()(conv16_sat)
        conv16_sat = Activation("relu")(conv16_sat)
        conv16_sat = Conv2D(512, (3, 3), padding="same", strides=1)(conv16_sat)
        conv16_sat = BatchNormalization()(conv16_sat)
        conv16_sat = Activation("relu")(conv16_sat)
        conv16_sat_add = Add()([conv16_sat, conv15_sat_add])
        conv16_sat_add = BatchNormalization()(conv16_sat_add)
        conv16_sat_add = Activation("relu")(conv16_sat_add)

        conv17_sat = Conv2D(512, (3, 3), padding="same", strides=1)(conv16_sat_add)
        conv17_sat = BatchNormalization()(conv17_sat)
        conv17_sat = Activation("relu")(conv17_sat)
        conv17_sat = Conv2D(512, (3, 3), padding="same", strides=1)(conv17_sat)
        conv17_sat = BatchNormalization()(conv17_sat)
        conv17_sat = Activation("relu")(conv17_sat)
        conv17_sat_add = Add()([conv17_sat, conv16_sat_add])
        conv17_sat_add = BatchNormalization()(conv17_sat_add)
        conv17_sat_add = Activation("relu")(conv17_sat_add)

        last_pool_sat = AveragePooling2D((2, 2), padding="same", strides=1)(conv17_sat_add)

        # satellite dilation8
        dilation8_sat = Conv2D(512, (3, 3), dilation_rate=1, padding="same")(last_pool_sat)
        dilation8_sat = Activation("relu")(dilation8_sat)
        dilation8_sat = Conv2D(512, (3, 3), dilation_rate=2, padding="same")(dilation8_sat)
        dilation8_sat = Activation("relu")(dilation8_sat)
        dilation8_sat = Conv2D(512, (3, 3), dilation_rate=4, padding="same")(dilation8_sat)
        dilation8_sat = Activation("relu")(dilation8_sat)
        dilation8_sat = Conv2D(512, (3, 3), dilation_rate=8, padding="same")(dilation8_sat)
        dilation8_sat = Activation("relu")(dilation8_sat)

        # satellite dilation4
        dilation4_sat = Conv2D(512, (3, 3), dilation_rate=1, padding="same")(last_pool_sat)
        dilation4_sat = Activation("relu")(dilation4_sat)
        dilation4_sat = Conv2D(512, (3, 3), dilation_rate=2, padding="same")(dilation4_sat)
        dilation4_sat = Activation("relu")(dilation4_sat)
        dilation4_sat = Conv2D(512, (3, 3), dilation_rate=4, padding="same")(dilation4_sat)
        dilation4_sat = Activation("relu")(dilation4_sat)

        # satellite dilation2
        dilation2_sat = Conv2D(512, (3, 3), dilation_rate=1, padding="same")(last_pool_sat)
        dilation2_sat = Activation("relu")(dilation2_sat)
        dilation2_sat = Conv2D(512, (3, 3), dilation_rate=2, padding="same")(dilation2_sat)
        dilation2_sat = Activation("relu")(dilation2_sat)

        # satellite dilation1
        dilation1_sat = Conv2D(512, (3, 3), dilation_rate=1, padding="same")(last_pool_sat)
        dilation1_sat = Activation("relu")(dilation1_sat)

        dilation_sat_out = Add()([last_pool_sat, dilation1_sat, dilation2_sat, dilation4_sat, dilation8_sat])

        # satellite decode level3
        decode3_sat = Conv2D(512, (1, 1), padding="same", strides=1)(dilation_sat_out)
        decode3_sat = BatchNormalization()(decode3_sat)
        decode3_sat = Activation("relu")(decode3_sat)

        decode3_sat = Conv2DTranspose(128, (3, 3), strides=2, padding="same")(decode3_sat)
        decode3_sat = BatchNormalization()(decode3_sat)
        decode3_sat = Activation("relu")(decode3_sat)

        decode3_sat = Conv2D(256, (1, 1), strides=1, padding="same")(decode3_sat)
        decode3_sat = BatchNormalization()(decode3_sat)
        decode3_sat = Activation("relu")(decode3_sat)

        decode3_sat_out = Add()([decode3_sat, conv14_sat_add])

        # satellite decode level2
        decode2_sat = Conv2D(256, (1, 1), padding="same", strides=1)(decode3_sat_out)
        decode2_sat = BatchNormalization()(decode2_sat)
        decode2_sat = Activation("relu")(decode2_sat)

        decode2_sat = Conv2DTranspose(64, (3, 3), strides=2, padding="same")(decode2_sat)
        decode2_sat = BatchNormalization()(decode2_sat)
        decode2_sat = Activation("relu")(decode2_sat)

        decode2_sat = Conv2D(128, (1, 1), strides=1, padding="same")(decode2_sat)
        decode2_sat = BatchNormalization()(decode2_sat)
        decode2_sat = Activation("relu")(decode2_sat)

        decode2_sat_out = Add()([decode2_sat, conv8_sat_add])

        # satellite decode level1
        decode1_sat = Conv2D(128, (1, 1), padding="same", strides=1)(decode2_sat_out)
        decode1_sat = BatchNormalization()(decode1_sat)
        decode1_sat = Activation("relu")(decode1_sat)

        decode1_sat = Conv2DTranspose(32, (3, 3), strides=2, padding="same")(decode1_sat)
        decode1_sat = BatchNormalization()(decode1_sat)
        decode1_sat = Activation("relu")(decode1_sat)

        decode1_sat = Conv2D(64, (1, 1), strides=1, padding="same")(decode1_sat)
        decode1_sat = BatchNormalization()(decode1_sat)
        decode1_sat = Activation("relu")(decode1_sat)

        # satellite decode out
        decode_sat_out = Add()([decode1_sat, conv4_sat_add])

        final_conv_sat = Conv2DTranspose(64, (3, 3), padding="same", strides=2)(decode_sat_out)
        final_conv_sat = Activation("relu")(final_conv_sat)
        final_conv_sat = Conv2D(32, (3, 3), padding="same")(final_conv_sat)
        final_conv_sat = Activation("relu")(final_conv_sat)

        # trajectory d-linknet stream
        conv1_traj = Conv2D(64, (7, 7), padding="same", strides=2)(input_traj)
        conv1_traj = BatchNormalization()(conv1_traj)
        conv1_traj = Activation("relu")(conv1_traj)
        conv1_traj = MaxPooling2D((3, 3), strides=2, padding="same")(conv1_traj)

        # trajectory level1
        conv2_traj = Conv2D(64, (3, 3), padding="same", strides=1)(conv1_traj)
        conv2_traj = BatchNormalization()(conv2_traj)
        conv2_traj = Activation("relu")(conv2_traj)
        conv2_traj = Conv2D(64, (3, 3), padding="same", strides=1)(conv2_traj)
        conv2_traj = BatchNormalization()(conv2_traj)
        conv2_traj = Activation("relu")(conv2_traj)
        conv2_traj_add = Add()([conv2_traj, conv1_traj])
        conv2_traj_add = BatchNormalization()(conv2_traj_add)
        conv2_traj_add = Activation("relu")(conv2_traj_add)

        conv3_traj = Conv2D(64, (3, 3), padding="same", strides=1)(conv2_traj_add)
        conv3_traj = BatchNormalization()(conv3_traj)
        conv3_traj = Activation("relu")(conv3_traj)
        conv3_traj = Conv2D(64, (3, 3), padding="same", strides=1)(conv3_traj)
        conv3_traj = BatchNormalization()(conv3_traj)
        conv3_traj = Activation("relu")(conv3_traj)
        conv3_traj_add = Add()([conv3_traj, conv2_traj_add])
        conv3_traj_add = BatchNormalization()(conv3_traj_add)
        conv3_traj_add = Activation("relu")(conv3_traj_add)

        conv4_traj = Conv2D(64, (3, 3), padding="same", strides=1)(conv3_traj_add)
        conv4_traj = BatchNormalization()(conv4_traj)
        conv4_traj = Activation("relu")(conv4_traj)
        conv4_traj = Conv2D(64, (3, 3), padding="same", strides=1)(conv4_traj)
        conv4_traj = BatchNormalization()(conv4_traj)
        conv4_traj = Activation("relu")(conv4_traj)
        conv4_traj_add = Add()([conv4_traj, conv3_traj_add])
        conv4_traj_add = BatchNormalization()(conv4_traj_add)
        conv4_traj_add = Activation("relu")(conv4_traj_add)

        # trajectory level2
        conv5_traj = Conv2D(128, (3, 3), padding="same", strides=2)(conv4_traj_add)
        conv5_traj = BatchNormalization()(conv5_traj)
        conv5_traj = Activation("relu")(conv5_traj)
        conv5_traj = Conv2D(128, (3, 3), padding="same", strides=1)(conv5_traj)
        conv5_traj = BatchNormalization()(conv5_traj)
        conv5_traj = Activation("relu")(conv5_traj)
        conv4_traj_add_skip = Conv2D(128, (1, 1), padding="same", strides=2)(conv4_traj_add)
        conv5_traj_add = Add()([conv5_traj, conv4_traj_add_skip])
        conv5_traj_add = BatchNormalization()(conv5_traj_add)
        conv5_traj_add = Activation("relu")(conv5_traj_add)

        conv6_traj = Conv2D(128, (3, 3), padding="same", strides=1)(conv5_traj_add)
        conv6_traj = BatchNormalization()(conv6_traj)
        conv6_traj = Activation("relu")(conv6_traj)
        conv6_traj = Conv2D(128, (3, 3), padding="same", strides=1)(conv6_traj)
        conv6_traj = BatchNormalization()(conv6_traj)
        conv6_traj = Activation("relu")(conv6_traj)
        conv6_traj_add = Add()([conv6_traj, conv5_traj_add])
        conv6_traj_add = BatchNormalization()(conv6_traj_add)
        conv6_traj_add = Activation("relu")(conv6_traj_add)

        conv7_traj = Conv2D(128, (3, 3), padding="same", strides=1)(conv6_traj_add)
        conv7_traj = BatchNormalization()(conv7_traj)
        conv7_traj = Activation("relu")(conv7_traj)
        conv7_traj = Conv2D(128, (3, 3), padding="same", strides=1)(conv7_traj)
        conv7_traj = BatchNormalization()(conv7_traj)
        conv7_traj = Activation("relu")(conv7_traj)
        conv7_traj_add = Add()([conv7_traj, conv6_traj_add])
        conv7_traj_add = BatchNormalization()(conv7_traj_add)
        conv7_traj_add = Activation("relu")(conv7_traj_add)

        conv8_traj = Conv2D(128, (3, 3), padding="same", strides=1)(conv7_traj_add)
        conv8_traj = BatchNormalization()(conv8_traj)
        conv8_traj = Activation("relu")(conv8_traj)
        conv8_traj = Conv2D(128, (3, 3), padding="same", strides=1)(conv8_traj)
        conv8_traj = BatchNormalization()(conv8_traj)
        conv8_traj = Activation("relu")(conv8_traj)
        conv8_traj_add = Add()([conv8_traj, conv7_traj_add])
        conv8_traj_add = BatchNormalization()(conv8_traj_add)
        conv8_traj_add = Activation("relu")(conv8_traj_add)

        # trajectory level3
        conv9_traj = Conv2D(256, (3, 3), padding="same", strides=2)(conv8_traj_add)
        conv9_traj = BatchNormalization()(conv9_traj)
        conv9_traj = Activation("relu")(conv9_traj)
        conv9_traj = Conv2D(256, (3, 3), padding="same", strides=1)(conv9_traj)
        conv9_traj = BatchNormalization()(conv9_traj)
        conv9_traj = Activation("relu")(conv9_traj)
        conv8_traj_add_skip = Conv2D(256, (1, 1), padding="same", strides=2)(conv8_traj_add)
        conv9_traj_add = Add()([conv9_traj, conv8_traj_add_skip])
        conv9_traj_add = BatchNormalization()(conv9_traj_add)
        conv9_traj_add = Activation("relu")(conv9_traj_add)

        conv10_traj = Conv2D(256, (3, 3), padding="same", strides=1)(conv9_traj_add)
        conv10_traj = BatchNormalization()(conv10_traj)
        conv10_traj = Activation("relu")(conv10_traj)
        conv10_traj = Conv2D(256, (3, 3), padding="same", strides=1)(conv10_traj)
        conv10_traj = BatchNormalization()(conv10_traj)
        conv10_traj = Activation("relu")(conv10_traj)
        conv10_traj_add = Add()([conv10_traj, conv9_traj_add])
        conv10_traj_add = BatchNormalization()(conv10_traj_add)
        conv10_traj_add = Activation("relu")(conv10_traj_add)

        conv11_traj = Conv2D(256, (3, 3), padding="same", strides=1)(conv10_traj_add)
        conv11_traj = BatchNormalization()(conv11_traj)
        conv11_traj = Activation("relu")(conv11_traj)
        conv11_traj = Conv2D(256, (3, 3), padding="same", strides=1)(conv11_traj)
        conv11_traj = BatchNormalization()(conv11_traj)
        conv11_traj = Activation("relu")(conv11_traj)
        conv11_traj_add = Add()([conv11_traj, conv10_traj_add])
        conv11_traj_add = BatchNormalization()(conv11_traj_add)
        conv11_traj_add = Activation("relu")(conv11_traj_add)

        conv12_traj = Conv2D(256, (3, 3), padding="same", strides=1)(conv11_traj_add)
        conv12_traj = BatchNormalization()(conv12_traj)
        conv12_traj = Activation("relu")(conv12_traj)
        conv12_traj = Conv2D(256, (3, 3), padding="same", strides=1)(conv12_traj)
        conv12_traj = BatchNormalization()(conv12_traj)
        conv12_traj = Activation("relu")(conv12_traj)
        conv12_traj_add = Add()([conv12_traj, conv11_traj_add])
        conv12_traj_add = BatchNormalization()(conv12_traj_add)
        conv12_traj_add = Activation("relu")(conv12_traj_add)

        conv13_traj = Conv2D(256, (3, 3), padding="same", strides=1)(conv12_traj_add)
        conv13_traj = BatchNormalization()(conv13_traj)
        conv13_traj = Activation("relu")(conv13_traj)
        conv13_traj = Conv2D(256, (3, 3), padding="same", strides=1)(conv13_traj)
        conv13_traj = BatchNormalization()(conv13_traj)
        conv13_traj = Activation("relu")(conv13_traj)
        conv13_traj_add = Add()([conv13_traj, conv12_traj_add])
        conv13_traj_add = BatchNormalization()(conv13_traj_add)
        conv13_traj_add = Activation("relu")(conv13_traj_add)

        conv14_traj = Conv2D(256, (3, 3), padding="same", strides=1)(conv13_traj_add)
        conv14_traj = BatchNormalization()(conv14_traj)
        conv14_traj = Activation("relu")(conv14_traj)
        conv14_traj = Conv2D(256, (3, 3), padding="same", strides=1)(conv14_traj)
        conv14_traj = BatchNormalization()(conv14_traj)
        conv14_traj = Activation("relu")(conv14_traj)
        conv14_traj_add = Add()([conv14_traj, conv13_traj_add])
        conv14_traj_add = BatchNormalization()(conv14_traj_add)
        conv14_traj_add = Activation("relu")(conv14_traj_add)

        # trajectory level4
        conv15_traj = Conv2D(512, (3, 3), padding="same", strides=2)(conv14_traj_add)
        conv15_traj = BatchNormalization()(conv15_traj)
        conv15_traj = Activation("relu")(conv15_traj)
        conv15_traj = Conv2D(512, (3, 3), padding="same", strides=1)(conv15_traj)
        conv15_traj = BatchNormalization()(conv15_traj)
        conv15_traj = Activation("relu")(conv15_traj)
        conv14_traj_add_skip = Conv2D(512, (1, 1), padding="same", strides=2)(conv14_traj_add)
        conv15_traj_add = Add()([conv15_traj, conv14_traj_add_skip])
        conv15_traj_add = BatchNormalization()(conv15_traj_add)
        conv15_traj_add = Activation("relu")(conv15_traj_add)

        conv16_traj = Conv2D(512, (3, 3), padding="same", strides=1)(conv15_traj_add)
        conv16_traj = BatchNormalization()(conv16_traj)
        conv16_traj = Activation("relu")(conv16_traj)
        conv16_traj = Conv2D(512, (3, 3), padding="same", strides=1)(conv16_traj)
        conv16_traj = BatchNormalization()(conv16_traj)
        conv16_traj = Activation("relu")(conv16_traj)
        conv16_traj_add = Add()([conv16_traj, conv15_traj_add])
        conv16_traj_add = BatchNormalization()(conv16_traj_add)
        conv16_traj_add = Activation("relu")(conv16_traj_add)

        conv17_traj = Conv2D(512, (3, 3), padding="same", strides=1)(conv16_traj_add)
        conv17_traj = BatchNormalization()(conv17_traj)
        conv17_traj = Activation("relu")(conv17_traj)
        conv17_traj = Conv2D(512, (3, 3), padding="same", strides=1)(conv17_traj)
        conv17_traj = BatchNormalization()(conv17_traj)
        conv17_traj = Activation("relu")(conv17_traj)
        conv17_traj_add = Add()([conv17_traj, conv16_traj_add])
        conv17_traj_add = BatchNormalization()(conv17_traj_add)
        conv17_traj_add = Activation("relu")(conv17_traj_add)

        last_pool_traj = AveragePooling2D((2, 2), padding="same", strides=1)(conv17_traj_add)

        # trajectory dilation8
        dilation8_traj = Conv2D(512, (3, 3), dilation_rate=1, padding="same")(last_pool_traj)
        dilation8_traj = Activation("relu")(dilation8_traj)
        dilation8_traj = Conv2D(512, (3, 3), dilation_rate=2, padding="same")(dilation8_traj)
        dilation8_traj = Activation("relu")(dilation8_traj)
        dilation8_traj = Conv2D(512, (3, 3), dilation_rate=4, padding="same")(dilation8_traj)
        dilation8_traj = Activation("relu")(dilation8_traj)
        dilation8_traj = Conv2D(512, (3, 3), dilation_rate=8, padding="same")(dilation8_traj)
        dilation8_traj = Activation("relu")(dilation8_traj)

        # trajectory dilation4
        dilation4_traj = Conv2D(512, (3, 3), dilation_rate=1, padding="same")(last_pool_traj)
        dilation4_traj = Activation("relu")(dilation4_traj)
        dilation4_traj = Conv2D(512, (3, 3), dilation_rate=2, padding="same")(dilation4_traj)
        dilation4_traj = Activation("relu")(dilation4_traj)
        dilation4_traj = Conv2D(512, (3, 3), dilation_rate=4, padding="same")(dilation4_traj)
        dilation4_traj = Activation("relu")(dilation4_traj)

        # trajectory dilation2
        dilation2_traj = Conv2D(512, (3, 3), dilation_rate=1, padding="same")(last_pool_traj)
        dilation2_traj = Activation("relu")(dilation2_traj)
        dilation2_traj = Conv2D(512, (3, 3), dilation_rate=2, padding="same")(dilation2_traj)
        dilation2_traj = Activation("relu")(dilation2_traj)

        # trajectory dilation1
        dilation1_traj = Conv2D(512, (3, 3), dilation_rate=1, padding="same")(last_pool_traj)
        dilation1_traj = Activation("relu")(dilation1_traj)

        dilation_traj_out = Add()([last_pool_traj, dilation1_traj, dilation2_traj, dilation4_traj, dilation8_traj])

        # trajectory decode level3
        decode3_traj = Conv2D(512, (1, 1), padding="same", strides=1)(dilation_traj_out)
        decode3_traj = BatchNormalization()(decode3_traj)
        decode3_traj = Activation("relu")(decode3_traj)

        decode3_traj = Conv2DTranspose(128, (3, 3), strides=2, padding="same")(decode3_traj)
        decode3_traj = BatchNormalization()(decode3_traj)
        decode3_traj = Activation("relu")(decode3_traj)

        decode3_traj = Conv2D(256, (1, 1), strides=1, padding="same")(decode3_traj)
        decode3_traj = BatchNormalization()(decode3_traj)
        decode3_traj = Activation("relu")(decode3_traj)

        decode3_traj_out = Add()([decode3_traj, conv14_traj_add])

        # trajectory decode level2
        decode2_traj = Conv2D(256, (1, 1), padding="same", strides=1)(decode3_traj_out)
        decode2_traj = BatchNormalization()(decode2_traj)
        decode2_traj = Activation("relu")(decode2_traj)

        decode2_traj = Conv2DTranspose(64, (3, 3), strides=2, padding="same")(decode2_traj)
        decode2_traj = BatchNormalization()(decode2_traj)
        decode2_traj = Activation("relu")(decode2_traj)

        decode2_traj = Conv2D(128, (1, 1), strides=1, padding="same")(decode2_traj)
        decode2_traj = BatchNormalization()(decode2_traj)
        decode2_traj = Activation("relu")(decode2_traj)

        decode2_traj_out = Add()([decode2_traj, conv8_traj_add])

        # trajectory decode level1
        decode1_traj = Conv2D(128, (1, 1), padding="same", strides=1)(decode2_traj_out)
        decode1_traj = BatchNormalization()(decode1_traj)
        decode1_traj = Activation("relu")(decode1_traj)

        decode1_traj = Conv2DTranspose(32, (3, 3), strides=2, padding="same")(decode1_traj)
        decode1_traj = BatchNormalization()(decode1_traj)
        decode1_traj = Activation("relu")(decode1_traj)

        decode1_traj = Conv2D(64, (1, 1), strides=1, padding="same")(decode1_traj)
        decode1_traj = BatchNormalization()(decode1_traj)
        decode1_traj = Activation("relu")(decode1_traj)

        # trajectory decode out
        decode_traj_out = Add()([decode1_traj, conv4_traj_add])

        final_conv_traj = Conv2DTranspose(64, (3, 3), padding="same", strides=2)(decode_traj_out)
        final_conv_traj = Activation("relu")(final_conv_traj)
        final_conv_traj = Conv2D(32, (3, 3), padding="same")(final_conv_traj)
        final_conv_traj = Activation("relu")(final_conv_traj)

        output_layer1 = Conv2D(2, (1, 1), padding="same", activation="sigmoid")(final_conv_sat)
        output_layer2 = Conv2D(2, (1, 1), padding="same", activation="sigmoid")(final_conv_traj)

        fusion_layer = get_fusion_layer(fusion_type, output_layer1, output_layer2)

        if fusion_type == "concat":
            fusion_layer = Conv2D(2, (1, 1), padding="same", activation="sigmoid")(fusion_layer)

        self.model = tf.keras.Model(inputs=inputs_layer, outputs=fusion_layer)

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_function,
                           metrics=get_metrics(batch_size=self.batch_size))
