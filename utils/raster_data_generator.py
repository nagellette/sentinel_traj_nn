'''
Modified version of https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
'''

import tensorflow as tf
import gdal
import numpy as np
from skimage.transform import rotate
from utils.raster_standardize import raster_standardize
from PIL import Image


class RasterDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, file_names,
                 file_path,
                 label_path,
                 generation_list,
                 batch_size=1,
                 dim=(572, 572),
                 shuffle=False,
                 ext="train",
                 save_image_file=None,
                 num_of_classes=2,
                 srcnn_count=0,
                 non_srcnn_count=False):
        '''
        Raster data generator for generating images from multiple raster datasets.

        :param file_names: list of files and data types
        :param file_path: work directory path
        :param label_path: label file name
        :param generation_list: pixel offset and augmentation values list
        :param batch_size: size of the batch
        :param dim: dimensions of the batch
        :param shuffle: shuffle Yes/No
        :param ext: extension for file save output extension
        :param save_image_file: image file save path, if not provided save ommited
        :param num_of_classes: number of classes for label dataset
        :param srcnn_count: count of raster layers to apply srcnn
        :param non_scrnn_count: count of raster layers to skip srcnn
        '''

        self.file_names = file_names
        self.file_path = file_path
        self.label_path = label_path
        self.generation_list = generation_list
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.ext = ext
        self.save_image_file = save_image_file
        self.num_of_classes = num_of_classes
        self.srcnn_count = srcnn_count
        self.non_scrnn_count = non_srcnn_count

        # TODO: remove after debug
        self.filename_counter = 0

        self.raster_files = []

        # filling raster files list with data type and min/max pixel values
        for file_name in self.file_names:
            temp_raster = gdal.Open(file_path + file_name[0])
            temp_raster_band = temp_raster.GetRasterBand(1)
            if temp_raster_band.GetMinimum() is None or temp_raster_band.GetMaximum() is None:
                print("Calculating band statistics: " + file_path + file_name[0])
                temp_raster_band.ComputeStatistics(0)
            temp_min = temp_raster_band.GetMinimum()
            temp_max = temp_raster_band.GetMaximum()
            temp_raster = None

            self.raster_files.append([gdal.Open(file_path + file_name[0]),
                                      file_name[1],
                                      temp_min,
                                      temp_max])

        self.label_raster = gdal.Open(file_path + label_path)

        self.on_epoch_end()

    def __len__(self):
        # number of batches per epoch
        return int(np.floor(len(self.generation_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.generation_list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.generation_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):

        train_batch_all = []
        label_batch = []
        counter = 0

        # filling batch dataset
        for list_ID in list_IDs_temp:
            train_batch = []
            for i, raster_file in enumerate(self.raster_files):

                raster_band = raster_file[0].GetRasterBand(1)

                # standardizing raster array values according to the data type
                raster_as_array = raster_standardize(
                    raster_band.ReadAsArray(list_ID[0], list_ID[1], self.dim[0], self.dim[1]),
                    raster_file[1],
                    raster_file[2],
                    raster_file[3])

                # applying data rotation augmentation
                if list_ID[2] != 0:
                    raster_as_array = rotate(raster_as_array, list_ID[2], cval=0.0)

                train_batch.append(raster_as_array)
                if self.save_image_file is not None:
                    raster_as_array = raster_as_array * 255
                    raster_as_array = raster_as_array.astype(np.uint8)
                    img = Image.fromarray(raster_as_array, 'L')
                    img.save(
                        self.save_image_file + str(self.filename_counter) + "_" +
                        self.ext + "_" +
                        str(list_ID[0]) +
                        "_" +
                        str(list_ID[1]) +
                        "-b" +
                        str(i) +
                        ".jpg")

            # fill data
            train_batch_all.append(train_batch)

            # fill label
            label_array = self.label_raster.GetRasterBand(1).ReadAsArray(list_ID[0], list_ID[1], self.dim[0],
                                                                         self.dim[1])

            if list_ID[2] != 0:
                # TODO: overcome the empty areas for the cases where rotation is not the exact product of 90
                label_array = rotate(label_array, list_ID[2], cval=0.0)

            label_array = tf.keras.utils.to_categorical(np.array(label_array), num_classes=self.num_of_classes)

            label_batch.append(label_array)
            if self.save_image_file is not None:
                for i in range(0, self.num_of_classes):
                    label_array_img = label_array[:, :, i] * 255.
                    label_array_img = label_array_img.astype(np.uint8)
                    img = Image.fromarray(label_array_img, 'L')
                    img.save(
                        self.save_image_file + str(self.filename_counter) + "_" +
                        self.ext +
                        "_" +
                        str(list_ID[0]) +
                        "_" +
                        str(list_ID[1]) +
                        "_label_" +
                        str(i) +
                        "_" +
                        str(counter) +
                        ".jpg")

            self.filename_counter += 1

        train_batch_all = np.array(train_batch_all)
        train_batch_all = np.rollaxis(train_batch_all, 2, 1)
        train_batch_all = np.rollaxis(train_batch_all, 3, 2)
        label_batch = np.array(label_batch)

        if self.srcnn_count == 0:
            # train_batch_all = np.reshape(train_batch_all, (self.batch_size, self.dim[0], self.dim[1], len(self.raster_files)), order='C')
            '''
            # debug start
            for i in range(len(self.raster_files)):
                raster_as_array = train_batch_all[0,:,:,i] * 255
                raster_as_array = raster_as_array.astype(np.uint8)
                img = Image.fromarray(raster_as_array, 'L')
                img.save(
                    "../debug/test/" +
                    self.ext + "_" + str(self.counter_temp) + "_" +
                     str(i) +
                    ".jpg")

            for i in range(0, 2):
                raster_as_array = label_batch[0,:,:,i] * 255
                raster_as_array = raster_as_array.astype(np.uint8)
                img = Image.fromarray(raster_as_array, 'L')
                img.save(
                    "../debug/test/" +
                    self.ext + "_" + str(self.counter_temp) + "_" +
                     str(i) + "_label" +
                    ".jpg")
            # debug end
            self.counter_temp += 1
            '''

            return train_batch_all, label_batch

        else:
            # TODO: add algorithm logic definition
            train_batch_srcnn = []
            for i in range(self.srcnn_count):
                temp_srcnn = train_batch_all[:, :, :, i]
                temp_srcnn = temp_srcnn.reshape([self.batch_size, self.dim[0], self.dim[1], 1])
                train_batch_srcnn.append(temp_srcnn)

            if self.non_scrnn_count:
                temp_srcnn = train_batch_all[:,
                             (len(self.raster_files) - self.srcnn_count):(len(self.raster_files) - 1), :, :]
                temp_srcnn = temp_srcnn.reshape(
                    [self.batch_size, self.dim[0], self.dim[1], len(self.raster_files) - self.srcnn_count])
                train_batch_srcnn.append(temp_srcnn)

            return train_batch_srcnn, label_batch
