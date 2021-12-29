try:
    import gdal
except:
    try:
        from osgeo import gdal
    except:
        print("GDAL cannot be imported.")

import sys
from utils.calc_augmentation_dim import calc_augmentation_dim
import numpy as np


class TrainTestValidateSplitter:
    def __init__(self,
                 sample_file,
                 mask_file=False,
                 image_index=0,
                 train=0,
                 test=0,
                 validation=0,
                 dim=0,
                 augment=0,
                 overlap=0,
                 seed=0,
                 check_coverage=False,
                 label_threshold=0,
                 check_label=False):

        # TODO: SAFELY remove mask based thresholding if possible.
        """
        Train, Test, Validation data list creator.

        :param sample_file: sample file for raster dimension template
        :param mask_file: mask file for masking unnecessary areas
        :param image_index: order of the processed image in all image set
        :param train: percentage of the train data in all data
        :param test: percentage of the test data in all data
        :param validation: percentage of the validation data in all data
        :param dim: dimensions of the batch
        :param augment: augmentation value in degrees
        :param overlap: percentage of the overlap between batch
        :param seed: if provided, use as seed value
        :param check_coverage: verify if the image mask contain at least %70 landmass. Added to avoid large areas
        :param label_threshold: threshold for label verification to tackle empty examples
        :param check_coverage: verify if the image label contain at least the given amount in threshold
        covered with sea. When True, sample file must be 0 and 1 mask image.
        """

        self.sample_file = sample_file
        self.mask_file = mask_file
        self.image_index = image_index
        self.train = train
        self.test = test
        self.validation = validation
        self.dim = dim
        # dim_initial added for future use of original dimensions
        self.dim_initial = dim
        self.augment = augment
        self.overlap = overlap
        self.seed = seed
        self.check_coverage = check_coverage
        self.label_threshold = label_threshold
        self.check_label = check_label

        raster_file = gdal.Open(sample_file)
        self.X = raster_file.RasterXSize
        self.Y = raster_file.RasterYSize

        self.raster = raster_file.GetRasterBand(1)

        if mask_file is not False:
            mask_raster_file = gdal.Open(mask_file)
            self.raster_mask = mask_raster_file.GetRasterBand(1)

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

                        # reading mask image and determining how much % of the pixels contain landmass.
                        coverage_percentage_mask = 0
                        if self.check_coverage:
                            raster_array_mask = self.raster_mask.ReadAsArray(self.dim[0] * x_i - shift,
                                                                             self.dim[1] * y_i - shift,
                                                                             new_dim,
                                                                             new_dim)

                            coverage_percentage_mask = np.sum(raster_array_mask) / (new_dim * new_dim)

                        # reading label image and determining how much % of the pixels contain label
                        coverage_percentage_label = 0
                        if self.check_label:
                            raster_array_label = self.raster.ReadAsArray(self.dim[0] * x_i - shift,
                                                                         self.dim[1] * y_i - shift,
                                                                         new_dim,
                                                                         new_dim)

                            coverage_percentage_label = np.sum(raster_array_label) / (new_dim * new_dim)

                        # add the tile to output data set if label coverage is more than label threshold
                        check_filters_label = False
                        if not self.check_label or coverage_percentage_label >= self.label_threshold:
                            check_filters_label = True

                        # add the tile to output data set if coverage is not being checked or coverage is more than %50
                        # and if the previous label check is passed
                        if (check_filters_label and self.check_coverage and coverage_percentage_mask >= 0.5) or (
                                check_filters_label and not self.check_coverage):
                            self.data_set.append([self.dim[0] * x_i,
                                                  self.dim[1] * y_i,
                                                  self.augment * augment_i,
                                                  new_dim,
                                                  shift,
                                                  coverage_percentage_label,
                                                  coverage_percentage_mask])

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
            train_list.append([self.data_set[file], self.image_index])

        for file in test_data:
            test_list.append([self.data_set[file], self.image_index])

        for file in validation_data:
            validation_list.append([self.data_set[file], self.image_index])

        # returning file lists
        return train_list, test_list, validation_list
