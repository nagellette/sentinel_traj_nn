import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, BatchNormalization
import sys
from tensorflow.keras.metrics import Recall, Precision, MeanIoU



class ModelRepository:
    def __init__(self, model_name, dim, input_channels, batch_size, srcnn_count=0):
        '''
        Collection of deep learning models for image segmentation.
        # TODO: add class parameter definitions
        :param model_name:
        :param dim:
        :param input_channels:
        :param batch_size:
        :param srcnn_count:
        '''
        self.model_name = model_name
        self.dim = dim
        self.input_channels = input_channels
        self.batch_size = batch_size
        self.srcnn_count = srcnn_count

        self.session_config = tf.compat.v1.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        self.session_config.gpu_options.per_process_gpu_memory_fraction = 0.90

        if self.model_name == "test_model":
            self.test_model()
        elif self.model_name == "unet":
            self.unet(self.dim, self.input_channels, self.batch_size)
        elif self.model_name == "srcnn_unet":
            self.srcnn_unet(self.dim, self.input_channels, self.batch_size, self.srcnn_count)
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
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def unet(self, dim, input_channels, batch_size):

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

        # TODO: add optimizer and loss function as model parameter
        opt = tf.keras.optimizers.SGD(lr=0.001,
                                      decay=1e-6,
                                      momentum=0.9,
                                      nesterov=True)

        self.model.compile(optimizer=opt,
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=["accuracy",
                                    Precision(),
                                    Recall(),
                                    MeanIoU(num_classes=2)])

    def srcnn_unet_4_alt(self, dim, input_channels, batch_size):

        # TODO: input channel dimensions are hard coded
        input_layer_1 = tf.keras.layers.Input((dim[0], dim[1], 1), batch_size=batch_size)
        srcnn1 = Conv2D(64, (9, 9), activation='relu', padding="same")(input_layer_1)
        srcnn1 = Conv2D(32, (1, 1), activation='relu', padding="same")(srcnn1)
        srcnn1 = Conv2D(1, (5, 5), activation='relu', padding="same")(srcnn1)

        input_layer_2 = tf.keras.layers.Input((dim[0], dim[1], 1), batch_size=batch_size)
        srcnn2 = Conv2D(64, (9, 9), activation='relu', padding="same")(input_layer_2)
        srcnn2 = Conv2D(32, (1, 1), activation='relu', padding="same")(srcnn2)
        srcnn2 = Conv2D(1, (5, 5), activation='relu', padding="same")(srcnn2)

        input_layer_3 = tf.keras.layers.Input((dim[0], dim[1], 1), batch_size=batch_size)
        srcnn3 = Conv2D(64, (9, 9), activation='relu', padding="same")(input_layer_3)
        srcnn3 = Conv2D(32, (1, 1), activation='relu', padding="same")(srcnn3)
        srcnn3 = Conv2D(1, (5, 5), activation='relu', padding="same")(srcnn3)

        input_layer_4 = tf.keras.layers.Input((dim[0], dim[1], 1), batch_size=batch_size)
        srcnn4 = Conv2D(64, (9, 9), activation='relu', padding="same")(input_layer_4)
        srcnn4 = Conv2D(32, (1, 1), activation='relu', padding="same")(srcnn4)
        srcnn4 = Conv2D(1, (5, 5), activation='relu', padding="same")(srcnn4)

        input_layer_5 = tf.keras.layers.Input((dim[0], dim[1], 3), batch_size=batch_size)

        main_inputs = concatenate([srcnn1, srcnn2, srcnn3, srcnn4, input_layer_5])

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

        self.model = tf.keras.Model(inputs=[input_layer_1,
                                            input_layer_2,
                                            input_layer_3,
                                            input_layer_4,
                                            input_layer_5], outputs=output_layer)

        # TODO: add optimizer, loss function and model metrics as model parameter
        opt = tf.keras.optimizers.SGD(lr=0.001,
                                      decay=1e-6,
                                      momentum=0.9,
                                      nesterov=True)

        self.model.compile(optimizer=opt,
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=["accuracy",
                                    Precision(),
                                    Recall(),
                                    MeanIoU(num_classes=2)])

    def srcnn_unet(self, dim, input_channels, batch_size, srcnn_count):

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

        # TODO: add optimizer, loss function and model metrics as model parameter
        opt = tf.keras.optimizers.SGD(lr=0.001,
                                      decay=1e-6,
                                      momentum=0.9,
                                      nesterov=True)

        self.model.compile(optimizer=opt,
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=["accuracy",
                                    Precision(),
                                    Recall(),
                                    MeanIoU(num_classes=2)])
