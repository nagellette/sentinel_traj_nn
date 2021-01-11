import sys
import tensorflow as tf
from models.model_repository import ModelRepository
from utils import raster_data_generator
from utils.input_reader import InputReader
from utils.train_test_splitter import TrainTestValidateSplitter
from utils.custom_callbacks import TimeKeeper
from datetime import datetime
import os
from PIL import Image
import numpy as np

tf.executing_eagerly()

# call out available GPU info
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# set start time variable for output file naming
start_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

# read input and configuration files
config = InputReader(sys.argv[1])
image_inputs = InputReader(sys.argv[2])

# set file paths from input file
work_directory, file_names, label_path = image_inputs.read_image()

# set model parameters from config file
BATCH_SIZE = config.get_batch_size()
IMAGE_DIMS = config.get_image_dim()
OPTIMIZER, L_RATE, DECAY, MOMENTUM, NESTEROV = config.get_optimizer_parameters()
EPOCH = config.get_epoch_count()
TRAIN_SIZE, TEST_SIZE, VALIDATE_SIZE = config.get_data_split()
AUGMENT = config.get_augmentation()
OVERLAP = config.get_overlap()
SHUFFLE = config.get_shuffle()
SEED = config.get_seed()
EPOCH_LIMIT = config.get_epoch_limit()
TEST_MODEL = config.get_test_model()
TEST_MODEL_LENGTH = config.get_test_model_count()
VALIDATION_MODEL_LENGTH = config.get_validation_model_count()
SRCNN_COUNT = config.get_srcnn_count()
OUTPUT_PATH = config.get_output_path()
LOSS = config.get_loss()

# splitting data into train, test and validation
data_splitter = TrainTestValidateSplitter(work_directory + label_path,
                                          TRAIN_SIZE,
                                          TEST_SIZE,
                                          VALIDATE_SIZE,
                                          IMAGE_DIMS,
                                          AUGMENT,
                                          OVERLAP,
                                          SEED)

# split data to train, test, validate
train_list, test_list, validation_list = data_splitter.get_train_test_validation()

# verify input values of test and validation sizes and convert to max validation/test data available if available
# data is short
if len(validation_list) < VALIDATION_MODEL_LENGTH:
    VALIDATION_MODEL_LENGTH = len(validation_list)
    print("Validation size converted to {}".format(len(validation_list)))

if len(test_list) < TEST_MODEL_LENGTH:
    TEST_MODEL_LENGTH = len(test_list)
    print("Test size converted to {}".format(len(test_list)))

print("Train data count:", str(len(train_list)))
print("Test data count:", str(len(test_list)))
print("Validation data count:", str(len(validation_list)))

# define output folder and create necessary folders
output_folder = OUTPUT_PATH + sys.argv[3] + "_" + start_time + "/"
image_outputs = output_folder + "images/"
os.system("mkdir " + output_folder)
os.system("mkdir " + image_outputs)

# saving model configuration and input files
os.system("cp " + sys.argv[1] + " " + output_folder + "config.json")
os.system("cp " + sys.argv[2] + " " + output_folder + "inputs.json")

# create data generators
train_data_generator = raster_data_generator.RasterDataGenerator(file_names=file_names,
                                                                 file_path=work_directory,
                                                                 label_path=label_path,
                                                                 generation_list=train_list,
                                                                 batch_size=BATCH_SIZE,
                                                                 dim=IMAGE_DIMS,
                                                                 shuffle=SHUFFLE,
                                                                 ext="train",
                                                                 srcnn_count=SRCNN_COUNT,
                                                                 non_srcnn_count=False)

validation_data_generator = raster_data_generator.RasterDataGenerator(file_names=file_names,
                                                                      file_path=work_directory,
                                                                      label_path=label_path,
                                                                      generation_list=validation_list,
                                                                      batch_size=BATCH_SIZE,
                                                                      dim=IMAGE_DIMS,
                                                                      shuffle=SHUFFLE,
                                                                      ext="val",
                                                                      srcnn_count=SRCNN_COUNT,
                                                                      non_srcnn_count=False)

test_data_generator = raster_data_generator.RasterDataGenerator(file_names=file_names,
                                                                file_path=work_directory,
                                                                label_path=label_path,
                                                                generation_list=test_list,
                                                                batch_size=1,
                                                                dim=IMAGE_DIMS,
                                                                shuffle=SHUFFLE,
                                                                ext="test",
                                                                save_image_file=image_outputs,
                                                                srcnn_count=SRCNN_COUNT,
                                                                non_srcnn_count=False)

# create model
model = ModelRepository(sys.argv[3],
                        IMAGE_DIMS,
                        len(file_names),
                        BATCH_SIZE,
                        srcnn_count=SRCNN_COUNT,
                        optimizer=OPTIMIZER,
                        l_rate=L_RATE,
                        decay=DECAY,
                        momentum=MOMENTUM,
                        nesterov=NESTEROV,
                        loss=LOSS).get_model()

# build model with defining input parameters
if sys.argv[3] == "unet":
    # build model with default provided dimensions for UNET
    model.build([IMAGE_DIMS[0], IMAGE_DIMS[1], len(file_names)])
elif sys.argv[3] == "srcnn_unet":
    build_input_dimensions = []
    # create input shapes which will run through SRCNN
    for i in range(0, SRCNN_COUNT):
        build_input_dimensions.append((IMAGE_DIMS[0], IMAGE_DIMS[1], 1))

    # create input shapes which won't run through SRCNN, if any
    if SRCNN_COUNT != len(file_names):
        build_input_dimensions.append((IMAGE_DIMS[0], IMAGE_DIMS[1], len(file_names) - SRCNN_COUNT))

    # build model
    model.build(build_input_dimensions)

print(model.summary())

# create csv logger callback
csv_logger = tf.keras.callbacks.CSVLogger(output_folder +
                                          sys.argv[3] +
                                          "_" +
                                          start_time +
                                          "_log.csv",
                                          append=False,
                                          separator=",")

# create reduce on plateau callback TODO: test with different options WARNING: Not used
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                 factor=0.1,
                                                 patience=5,
                                                 min_lr=0.0001)

# model checkpoint callback for saving best achieved weights
checkpoint = tf.keras.callbacks.ModelCheckpoint(output_folder +
                                                sys.argv[3] +
                                                "_" +
                                                start_time +
                                                "_checkpoint",
                                                monitor='val_loss',
                                                save_best_only=True,
                                                save_freq="epoch")

# model train/test/validation time keeper by epoch and iteration
time_keeper = TimeKeeper(log_path=output_folder)

# train model
history = model.fit_generator(train_data_generator,
                              validation_data=validation_data_generator,
                              epochs=EPOCH,
                              shuffle=SHUFFLE,
                              callbacks=[csv_logger, checkpoint, time_keeper],
                              steps_per_epoch=EPOCH_LIMIT,
                              validation_steps=VALIDATION_MODEL_LENGTH)

# mark output log file as complete if train succeeded
os.system("mv " + output_folder +
          sys.argv[3] +
          "_" + start_time +
          "_log.csv " + output_folder +
          sys.argv[3] +
          "_" +
          start_time
          + "_log_completed.csv ")

# save model
model.save(output_folder +
           sys.argv[3] +
           "_" +
           start_time +
           "_model")

if TEST_MODEL:
    print("Creating test output:")
    predictions = model.predict_generator(test_data_generator, steps=TEST_MODEL_LENGTH)

    i = predictions.shape
    print("shape of predictions")
    print(i)
    i = i[0]
    for j in range(i):
        img = predictions[j, :, :, 0] * 255.
        img = img.astype(np.uint8)
        img = Image.fromarray(img, 'L')
        img.save(image_outputs + "/" + str(j) + "_predict_" + str(j) + "_1.png")

        img = predictions[j, :, :, 1] * 255.
        img = img.astype(np.uint8)
        img = Image.fromarray(img, 'L')
        img.save(image_outputs + "/" + str(j) + "_predict_" + str(j) + "_2.png")
