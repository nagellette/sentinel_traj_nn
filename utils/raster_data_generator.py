import tensorflow as tf

try:
    import gdal
except:
    try:
        from osgeo import gdal
    except:
        print("GDAL cannot be imported.")
import numpy as np
from skimage.transform import rotate
from utils.raster_standardize import raster_standardize
from PIL import Image
from utils.get_file_extension import get_file_extension


class RasterDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, inputs,
                 generation_list,
                 batch_size=1,
                 dim=(572, 572),
                 shuffle=False,
                 ext="train",
                 save_image_file=None,
                 num_of_classes=2,
                 srcnn_count=0,
                 non_srcnn_count=False):
        """
        Raster data generator for generating images from multiple raster datasets.

        :param inputs: file_names, file_path and label_path list of files and data types
        :param generation_list: pixel offset and augmentation values list
        :param batch_size: size of the batch
        :param dim: dimensions of the batch
        :param shuffle: shuffle Yes/No
        :param ext: extension for file save output extension
        :param save_image_file: image file save path, if not provided save omitted
        :param num_of_classes: number of classes for label dataset
        :param srcnn_count: count of raster layers to apply srcnn
        :param non_srcnn_count: count of raster layers to skip srcnn
        """

        self.inputs = inputs
        self.generation_list = generation_list
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.ext = ext
        self.save_image_file = save_image_file
        self.num_of_classes = num_of_classes
        self.srcnn_count = srcnn_count
        self.non_scrnn_count = non_srcnn_count
        self.filename_counter = 0

        self.raster_files = []
        self.label_raster = []

        # run init for each input set, add to raster and label lists
        for _input in inputs:
            file_names = _input[1]
            file_path = _input[0]
            label_path = _input[2]

            # filling raster files list with data type and min/max pixel values
            raster_files_temp = []
            for file_name in file_names:
                temp_raster = gdal.Open(file_path + file_name[0])
                temp_raster_band = temp_raster.GetRasterBand(1)
                if temp_raster_band.GetMinimum() is None or temp_raster_band.GetMaximum() is None:
                    print("Calculating band statistics: " + file_path + file_name[0])
                    temp_raster_band.ComputeStatistics(0)
                temp_min = file_name[2]
                temp_max = file_name[3]
                temp_raster = None

                raster_files_temp.append([gdal.Open(file_path + file_name[0]),
                                          file_name[1],
                                          temp_min,
                                          temp_max,
                                          file_name[0]])

            self.raster_files.append(raster_files_temp)
            self.label_raster.append(gdal.Open(file_path + label_path))

        self.on_epoch_end()

    def __len__(self):
        # number of batches per epoch
        return int(np.floor(len(self.generation_list) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.generation_list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.generation_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):

        train_batch_all = []
        label_batch = []
        counter = 0

        # filling batch dataset
        for data in list_IDs_temp:
            train_batch = []

            # assign variable values by input set from input list
            list_ID = data[0]
            data_index = data[1]
            rasters = self.raster_files[data_index]

            for i, raster in enumerate(rasters):

                raster_band = raster[0].GetRasterBand(1)

                # standardizing raster array values according to the data type
                raster_as_array = raster_standardize(
                    raster_band.ReadAsArray(list_ID[0] - list_ID[4], list_ID[1] - list_ID[4], list_ID[3], list_ID[3]),
                    raster[1],
                    raster[2],
                    raster[3])

                # applying data rotation augmentation
                if list_ID[2] != 0:
                    raster_as_array = rotate(raster_as_array, list_ID[2], cval=0.0)
                    if list_ID[2] % 90 != 0:
                        # cropping image to model input dimensions
                        raster_as_array = raster_as_array[list_ID[4]:-list_ID[4], list_ID[4]:-list_ID[4]]

                train_batch.append(raster_as_array)
                # image save if save image set to True - file name counter added to the saved file name with
                # file name counter.
                if self.save_image_file is not None:
                    raster_as_array = raster_as_array * 255
                    raster_as_array = raster_as_array.astype(np.uint8)
                    img = Image.fromarray(raster_as_array, 'L')
                    img.save(
                        self.save_image_file + str(self.filename_counter) + "_" + str(data_index) + "_" +
                        self.ext + "_" +
                        str(list_ID[0]) +
                        "_" +

                        str(list_ID[1]) +
                        "_" +
                        str(list_ID[2]) +
                        get_file_extension(raster[4]) +
                        ".jpg")

            # fill data to main batch
            train_batch_all.append(train_batch)

            # fill label
            label_array = self.label_raster[data_index].GetRasterBand(1).ReadAsArray(list_ID[0] - list_ID[4],
                                                                                     list_ID[1] - list_ID[4],
                                                                                     list_ID[3],
                                                                                     list_ID[3])

            # applying data rotation augmentation to label image
            if list_ID[2] != 0:
                label_array = rotate(label_array, list_ID[2], cval=0.0, preserve_range=True)
                if list_ID[2] % 90 != 0:
                    # cropping image to model input dimensions
                    label_array = label_array[list_ID[4]:-list_ID[4], list_ID[4]:-list_ID[4]]

            # create categorical labels, one layer per class. two layers created if labels binary.
            label_array = tf.keras.utils.to_categorical(np.array(label_array), num_classes=self.num_of_classes)
            label_batch.append(label_array)

            # image save if save image set to True - file name counter added to the saved file name with
            # file name counter
            if self.save_image_file is not None:
                for i in range(0, self.num_of_classes):
                    label_array_img = label_array[:, :, i] * 255.
                    label_array_img = label_array_img.astype(np.uint8)
                    img = Image.fromarray(label_array_img, 'L')
                    img.save(
                        self.save_image_file + str(self.filename_counter) + "_" + str(data_index) + "_" +
                        self.ext +
                        "_" +
                        str(list_ID[0]) +
                        "_" +
                        str(list_ID[1]) +
                        "_" +
                        str(list_ID[2]) +
                        "_label_" +
                        str(i) +
                        "_" +
                        str(counter) +
                        ".jpg")

            # incrementing file name counter
            self.filename_counter += 1

        # converting input to model input data dimensions and data formats
        train_batch_all = np.array(train_batch_all)
        train_batch_all = np.rollaxis(train_batch_all, 2,
                                      1)  # converting from "bands first" tensor to "bands last" tensor
        train_batch_all = np.rollaxis(train_batch_all, 3,
                                      2)  # converting from "bands first" tensor to "bands last" tensor, continued
        label_batch = np.array(label_batch)

        if self.srcnn_count == 0:
            # return as raw if SRCNN is not used.
            return train_batch_all, label_batch

        else:
            # custom return once SRCNN is used.
            train_batch_srcnn = []

            # fill in the layers one by one which will run through SRCNN
            for i in range(self.srcnn_count):
                temp_srcnn = train_batch_all[:, :, :, i]
                temp_srcnn = temp_srcnn.reshape([self.batch_size, self.dim[0], self.dim[1], 1])
                train_batch_srcnn.append(temp_srcnn)

            # create one stack of layers which will skip SRCNN TODO: not tested. check outputs dimensions and actual
            #  outputs
            if self.non_scrnn_count:
                temp_srcnn = train_batch_all[:,
                             (len(self.raster_files) - self.srcnn_count):(len(self.raster_files) - 1), :, :]
                temp_srcnn = temp_srcnn.reshape(
                    [self.batch_size, self.dim[0], self.dim[1], len(self.raster_files) - self.srcnn_count])
                train_batch_srcnn.append(temp_srcnn)

            return train_batch_srcnn, label_batch
