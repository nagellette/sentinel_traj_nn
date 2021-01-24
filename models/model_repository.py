import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, \
    Activation, Add, Concatenate, UpSampling2D
import sys
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
        # level1
        conv1 = Conv2D(64, (3, 3), padding="same", strides=1)(inputs_layer)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation("relu")(conv1)
        conv1 = Conv2D(64, (3, 3), padding="same", strides=1)(conv1)

        conv1_shortcut = Conv2D(64, (1, 1), padding="same", strides=1)(inputs_layer)
        conv1_shortcut = BatchNormalization()(conv1_shortcut)
        conv1_shortcut = Activation("relu")(conv1_shortcut)

        conv1_output = Add()([conv1, conv1_shortcut])

        # level2
        conv2 = BatchNormalization()(conv1_output)
        conv2 = Activation("relu")(conv2)
        conv2 = Conv2D(128, (3, 3), padding="same", strides=2)(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation("relu")(conv2)
        conv2 = Conv2D(128, (3, 3), padding="same", strides=1)(conv2)

        conv2_shortcut = Conv2D(128, (1, 1), padding="same", strides=2)(conv1_output)
        conv2_shortcut = BatchNormalization()(conv2_shortcut)
        conv2_shortcut = Activation("relu")(conv2_shortcut)

        conv2_output = Add()([conv2, conv2_shortcut])

        # level3
        conv3 = BatchNormalization()(conv2_output)
        conv3 = Activation("relu")(conv3)
        conv3 = Conv2D(256, (3, 3), padding="same", strides=2)(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation("relu")(conv3)
        conv3 = Conv2D(256, (3, 3), padding="same", strides=1)(conv3)

        conv3_shortcut = Conv2D(256, (1, 1), padding="same", strides=2)(conv2_output)
        conv3_shortcut = BatchNormalization()(conv3_shortcut)
        conv3_shortcut = Activation("relu")(conv3_shortcut)

        conv3_output = Add()([conv3, conv3_shortcut])

        # level4
        conv4 = BatchNormalization()(conv3_output)
        conv4 = Activation("relu")(conv4)
        conv4 = Conv2D(512, (3, 3), padding="same", strides=2)(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation("relu")(conv4)
        conv4 = Conv2D(512, (3, 3), padding="same", strides=1)(conv4)

        conv4_shortcut = Conv2D(512, (1, 1), padding="same", strides=2)(conv3_output)
        conv4_shortcut = BatchNormalization()(conv4_shortcut)
        conv4_shortcut = Activation("relu")(conv4_shortcut)

        conv4_output = Add()([conv4, conv4_shortcut])

        # level5
        conv5 = BatchNormalization()(conv4_output)
        conv5 = Activation("relu")(conv5)
        conv5 = Conv2D(1024, (3, 3), padding="same", strides=2)(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation("relu")(conv5)
        conv5 = Conv2D(1024, (3, 3), padding="same", strides=1)(conv5)

        conv5_shortcut = Conv2D(1024, (1, 1), padding="same", strides=2)(conv4_output)
        conv5_shortcut = BatchNormalization()(conv5_shortcut)
        conv5_shortcut = Activation("relu")(conv5_shortcut)

        conv5_output = Add()([conv5, conv5_shortcut])

        # level6
        upscale6 = UpSampling2D((2, 2))(conv5_output)
        conc6 = Concatenate()([upscale6, conv4_output])

        conv6 = BatchNormalization()(conc6)
        conv6 = Activation("relu")(conv6)
        conv6 = Conv2D(512, (3, 3), padding="same", strides=1)(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Activation("relu")(conv6)
        conv6 = Conv2D(512, (3, 3), padding="same", strides=1)(conv6)

        conv6_shortcut = Conv2D(512, (1, 1), padding="same", strides=1)(conc6)
        conv6_shortcut = BatchNormalization()(conv6_shortcut)
        conv6_shortcut = Activation("relu")(conv6_shortcut)

        conv6_output = Add()([conv6, conv6_shortcut])

        # level7
        upscale7 = UpSampling2D((2, 2))(conv6_output)
        conc7 = Concatenate()([upscale7, conv3_output])

        conv7 = BatchNormalization()(conc7)
        conv7 = Activation("relu")(conv7)
        conv7 = Conv2D(256, (3, 3), padding="same", strides=1)(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Activation("relu")(conv7)
        conv7 = Conv2D(256, (3, 3), padding="same", strides=1)(conv7)

        conv7_shortcut = Conv2D(256, (1, 1), padding="same", strides=1)(conc7)
        conv7_shortcut = BatchNormalization()(conv7_shortcut)
        conv7_shortcut = Activation("relu")(conv7_shortcut)

        conv7_output = Add()([conv7, conv7_shortcut])

        # level8
        upscale8 = UpSampling2D((2, 2))(conv7_output)
        conc8 = Concatenate()([upscale8, conv2_output])

        conv8 = BatchNormalization()(conc8)
        conv8 = Activation("relu")(conv8)
        conv8 = Conv2D(128, (3, 3), padding="same", strides=1)(conv8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Activation("relu")(conv8)
        conv8 = Conv2D(128, (3, 3), padding="same", strides=1)(conv8)

        conv8_shortcut = Conv2D(128, (1, 1), padding="same", strides=1)(conc8)
        conv8_shortcut = BatchNormalization()(conv8_shortcut)
        conv8_shortcut = Activation("relu")(conv8_shortcut)

        conv8_output = Add()([conv8, conv8_shortcut])

        # level9
        upscale9 = UpSampling2D((2, 2))(conv8_output)
        conc9 = Concatenate()([upscale9, conv1_output])

        conv9 = BatchNormalization()(conc9)
        conv9 = Activation("relu")(conv9)
        conv9 = Conv2D(64, (3, 3), padding="same", strides=1)(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Activation("relu")(conv9)
        conv9 = Conv2D(64, (3, 3), padding="same", strides=1)(conv9)

        conv9_shortcut = Conv2D(64, (1, 1), padding="same", strides=1)(conc9)
        conv9_shortcut = BatchNormalization()(conv9_shortcut)
        conv9_shortcut = Activation("relu")(conv9_shortcut)

        conv9_output = Add()([conv9, conv9_shortcut])

        output_layer = Conv2D(2, (1, 1), padding="same", activation="sigmoid")(conv9_output)

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
        # level1
        conv1 = Conv2D(64, (3, 3), padding="same", strides=(1, 1))(inputs_layer)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation("relu")(conv1)
        conv1 = Conv2D(64, (3, 3), padding="same", strides=(1, 1))(conv1)

        conv1_shortcut = Conv2D(64, (1, 1), padding="same", strides=(1, 1))(inputs_layer)

        conv1_output = Add()([conv1, conv1_shortcut])

        # level2
        conv2 = BatchNormalization()(conv1_output)
        conv2 = Activation("relu")(conv2)
        conv2 = Conv2D(128, (3, 3), padding="same", strides=(2, 2))(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation("relu")(conv2)
        conv2 = Conv2D(128, (3, 3), padding="same", strides=(1, 1))(conv2)

        conv2_shortcut = Conv2D(128, (1, 1), padding="same", strides=(2, 2))(conv1_output)

        conv2_output = Add()([conv2, conv2_shortcut])

        # level3
        conv3 = BatchNormalization()(conv2_output)
        conv3 = Activation("relu")(conv3)
        conv3 = Conv2D(256, (3, 3), padding="same", strides=(2, 2))(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation("relu")(conv3)
        conv3 = Conv2D(256, (3, 3), padding="same", strides=(1, 1))(conv3)

        conv3_shortcut = Conv2D(256, (1, 1), padding="same", strides=(2, 2))(conv2_output)

        conv3_output = Add()([conv3, conv3_shortcut])

        # level5
        conv5 = BatchNormalization()(conv3_output)
        conv5 = Activation("relu")(conv5)
        conv5 = Conv2D(512, (3, 3), padding="same", strides=(2, 2))(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation("relu")(conv5)
        conv5 = Conv2D(512, (3, 3), padding="same", strides=(1, 1))(conv5)

        conv5_shortcut = Conv2D(512, (1, 1), padding="same", strides=(2, 2))(conv3_output)

        conv5_output = Add()([conv5, conv5_shortcut])

        # level7
        upscale7 = UpSampling2D((2, 2))(conv5_output)
        conc7 = Concatenate()([upscale7, conv3_output])

        conv7 = BatchNormalization()(conc7)
        conv7 = Activation("relu")(conv7)
        conv7 = Conv2D(256, (3, 3), padding="same", strides=(1, 1))(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Activation("relu")(conv7)
        conv7 = Conv2D(256, (3, 3), padding="same", strides=(1, 1))(conv7)

        conv7_shortcut = Conv2D(256, (1, 1), padding="same", strides=(1, 1))(conc7)

        conv7_output = Add()([conv7, conv7_shortcut])

        # level8
        upscale8 = UpSampling2D((2, 2))(conv7_output)
        conc8 = Concatenate()([upscale8, conv2_output])

        conv8 = BatchNormalization()(conc8)
        conv8 = Activation("relu")(conv8)
        conv8 = Conv2D(128, (3, 3), padding="same", strides=(1, 1))(conv8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Activation("relu")(conv8)
        conv8 = Conv2D(128, (3, 3), padding="same", strides=(1, 1))(conv8)

        conv8_shortcut = Conv2D(128, (1, 1), padding="same", strides=(1, 1))(conc8)

        conv8_output = Add()([conv8, conv8_shortcut])

        # level9
        upscale9 = UpSampling2D((2, 2))(conv8_output)
        conc9 = Concatenate()([upscale9, conv1_output])

        conv9 = BatchNormalization()(conc9)
        conv9 = Activation("relu")(conv9)
        conv9 = Conv2D(64, (3, 3), padding="same", strides=(1, 1))(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Activation("relu")(conv9)
        conv9 = Conv2D(64, (3, 3), padding="same", strides=(1, 1))(conv9)

        conv9_shortcut = Conv2D(64, (1, 1), padding="same", strides=(1, 1))(conc9)

        conv9_output = Add()([conv9, conv9_shortcut])

        output_layer = Conv2D(2, (1, 1), padding="same", activation="sigmoid")(conv9_output)

        self.model = tf.keras.Model(inputs=inputs_layer, outputs=output_layer)

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_function,
                           metrics=get_metrics())
