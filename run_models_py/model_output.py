import tensorflow as tf
from utils import raster_data_generator
from utils.train_test_splitter import TrainTestValidateSplitter
from utils.input_reader import InputReader
import sys
from skimage.io import imsave

model = tf.keras.models.load_model("../run_models_py/output_models/unet_2020-04-21-11:07:27", compile=True)

# read input and configuration files
config = InputReader(sys.argv[1])
image_inputs = InputReader(sys.argv[2])

# set file paths from input file
work_directory, file_names, label_path = image_inputs.read_image()

# set model parameters from config file
BATCH_SIZE = config.get_batch_size()
IMAGE_DIMS = config.get_image_dim()
L_RATE = config.get_learning_rate()
EPOCH = config.get_epoch_count()
TRAIN_SIZE, TEST_SIZE, VALIDATE_SIZE = config.get_data_split()
AUGMENT = config.get_augmentation()
OVERLAP = config.get_overlap()
SHUFFLE = config.get_shuffle()
SEED = config.get_seed()
EPOCH_LIMIT = config.get_epoch_limit()

# splitting data into train, test and validation
data_splitter = TrainTestValidateSplitter(work_directory + label_path, TRAIN_SIZE, TEST_SIZE, VALIDATE_SIZE, IMAGE_DIMS,
                                          AUGMENT, OVERLAP)

# train, test, validate = data_splitter._data_sets()
train_list, test_list, validation_list = data_splitter.get_train_test_validation()

test_data_generator = raster_data_generator.RasterDataGenerator(file_names=file_names,
                                                                file_path=work_directory,
                                                                label_path=label_path,
                                                                generation_list=test_list,
                                                                batch_size=BATCH_SIZE,
                                                                dim=IMAGE_DIMS,
                                                                shuffle=SHUFFLE,
                                                                ext="test")

predictions = model.predict_generator(test_data_generator, steps=10)

i = predictions.shape
i = i[0]
for j in range(i):
    imsave("../debug/images/predict_" + str(j) + "_1.png", predictions[j,:,:,0])
    imsave("../debug/images/predict_" + str(j) + "_2.png", predictions[j,:,:,1])