import numpy as np
try:
    import gdal
except:
    try:
        from osgeo import gdal
    except:
        print("GDAL cannot be imported.")
import sys
from utils.calc_augmentation_dim import calc_augmentation_dim


class TrainTestValidateSplitter:
    def __init__(self, sample_file, train, test, validation, dim, augment=0, overlap=0, seed=0):
        """
        Train, Test, Validation data list creator.

        :param sample_file: sample file for raster dimension template
        :param train: percentage of the train data in all data
        :param test: percentage of the test data in all data
        :param validation: percentage of the validation data in all data
        :param dim: dimensions of the batch
        :param augment: augmentation value in degrees
        :param overlap: percentage of the overlap between batch
        :param seed: if provided, use as seed value
        """

        self.sample_file = sample_file
        self.train = train
        self.test = test
        self.validation = validation
        self.dim = dim
        # dim_initial added for future use of original dimensions
        self.dim_initial = dim
        self.augment = augment
        self.overlap = overlap
        self.seed = seed

        raster_file = gdal.Open(sample_file)
        self.X = raster_file.RasterXSize
        self.Y = raster_file.RasterYSize

        # calculating dimensions if there is an overlap
        if self.overlap != 0:
            if self.overlap < 1:
                new_dimx = np.floor(self.dim[0] * (1 - self.overlap))
                new_dimy = np.floor(self.dim[1] * (1 - self.overlap))
                self.dim = (new_dimx, new_dimy)
            else:
                sys.exit("Overlap value should be smaller than 1.")

        # checking augmentation option
        if self.augment > 180:
            sys.exit("Augmentation can not be bigger than 180.")

        self.data_set = []

        # initiating dataset list creation
        self.create_data_sets()

    def create_data_sets(self):
        x_count = int(np.floor(self.X / self.dim[0])) - 1
        y_count = int(np.floor(self.Y / self.dim[1])) - 1
        if self.augment != 0:
            augment_count = int(np.floor(360.0 / self.augment))
        else:
            augment_count = 1

        for x_i in range(0, x_count):
            for y_i in range(0, y_count):
                for augment_i in range(0, augment_count):
                    new_dim, shift = calc_augmentation_dim(self.dim_initial[0], self.augment * augment_i)
                    # check if the new dimensions are out of main image dimensions before adding image info to main list
                    if (self.dim[0] * x_i) - shift >= 0 and \
                            (self.dim[1] * y_i) - shift >= 0 and \
                            (self.dim[0] * x_i) + new_dim <= self.X and \
                            (self.dim[1] * y_i) + new_dim <= self.Y:
                        self.data_set.append([self.dim[0] * x_i,
                                              self.dim[1] * y_i,
                                              self.augment * augment_i,
                                              new_dim,
                                              shift])

    def get_train_test_validation(self):
        # get list of files and generate numbered list
        input_set_ID_list = np.arange(len(self.data_set))

        # shuffle the list
        if self.seed is not None:
            np.random.seed(self.seed)
            np.random.shuffle(input_set_ID_list)

        # create train, test boundaries as integer
        train_count = int(self.train * len(self.data_set))
        test_count = int(self.test * len(self.data_set)) + train_count

        # get train, test, validation data split file indexes
        train_data, test_data, validation_data = np.split(input_set_ID_list, [train_count, test_count])

        # creating empty lists to fill
        train_list = []
        test_list = []
        validation_list = []

        # filling  train, test, validation lists with file names
        for file in train_data:
            train_list.append(self.data_set[file])

        for file in test_data:
            test_list.append(self.data_set[file])

        for file in validation_data:
            validation_list.append(self.data_set[file])

        # returning file lists
        return train_list, test_list, validation_list
