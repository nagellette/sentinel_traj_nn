'''
Modified version of https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
'''

import tensorflow as tf
import gdal
import numpy as np
from skimage.transform import rotate
from utils.raster_standardize import raster_standardize
from skimage.io import imsave


class RasterDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, file_names, file_path, label_path, generation_list, batch_size=1, dim=(572, 572), shuffle=False,
                 ext="train"):
        '''
        :param file_names: list of files and data types
        :param file_path: work directory path
        :param label_path: label file name
        :param generation_list: pixel offset and augmentation values list
        :param batch_size: size of the batch
        :param dim: dimensions of the batch
        :param shuffle: shuffle Yes/No
        '''
        self.file_names = file_names
        self.file_path = file_path
        self.label_path = label_path
        self.generation_list = generation_list
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.ext = ext

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

                if i == 1:
                    raster_as_array = raster_as_array * 255
                    raster_as_array = raster_as_array.astype(np.uint8)

                    imsave("../debug/images/" + self.ext + "_" + str(list_ID[0]) + "_" + str(list_ID[1]) + ""
                                                                                                          "-b2.png",
                           raster_as_array)

                elif i == 13:
                    raster_as_array = raster_as_array * 200
                    raster_as_array = raster_as_array.astype(np.uint8)

                    imsave(
                        "../debug/images/" + self.ext + "_" + str(list_ID[0]) + "_" + str(list_ID[1]) + "_b_speed.png",
                        raster_as_array)


            # fill data
            train_batch_all.append(train_batch)

            # fill label
            label_array = self.label_raster.GetRasterBand(1).ReadAsArray(list_ID[0], list_ID[1], self.dim[0],
                                                                         self.dim[1])

            if list_ID[2] != 0:
                label_array = rotate(label_array, list_ID[2], cval=0.0)

            label_array = tf.keras.utils.to_categorical(np.array(label_array), num_classes=2)

            label_batch.append(label_array)

            imsave("../debug/images/" + self.ext + "_" + str(list_ID[0]) + "_" + str(list_ID[1]) + "_label_1_" + str(counter) + ".png",
                   label_array[:,:,0])

            imsave("../debug/images/" + self.ext + "_" + str(list_ID[0]) + "_" + str(list_ID[1]) + "_label_2_" + str(counter) + ".png",
                   label_array[:,:,1])
            counter += 1

        train_batch_all = np.array(train_batch_all)
        label_batch = np.array(label_batch)

        return train_batch_all.reshape([self.batch_size, self.dim[0], self.dim[1], len(self.raster_files)]), label_batch
