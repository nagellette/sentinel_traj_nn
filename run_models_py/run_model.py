import sys
import tensorflow as tf
from models.model_repository import ModelRepository
from utils import raster_data_generator
from utils.input_reader import InputReader
from utils.train_test_splitter import TrainTestValidateSplitter
from datetime import datetime
import os

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

print("Train data count:", str(len(train_list)))
print("Test data count:", str(len(test_list)))
print("Validation data count:", str(len(validation_list)))

train_data_generator = raster_data_generator.RasterDataGenerator(file_names=file_names,
                                                                 file_path=work_directory,
                                                                 label_path=label_path,
                                                                 generation_list=train_list,
                                                                 batch_size=BATCH_SIZE,
                                                                 dim=IMAGE_DIMS,
                                                                 shuffle=SHUFFLE,
                                                                 ext="train")

test_data_generator = raster_data_generator.RasterDataGenerator(file_names=file_names,
                                                                file_path=work_directory,
                                                                label_path=label_path,
                                                                generation_list=test_list,
                                                                batch_size=BATCH_SIZE,
                                                                dim=IMAGE_DIMS,
                                                                shuffle=SHUFFLE,
                                                                ext="test")

validation_data_generator = raster_data_generator.RasterDataGenerator(file_names=file_names,
                                                                      file_path=work_directory,
                                                                      label_path=label_path,
                                                                      generation_list=validation_list,
                                                                      batch_size=BATCH_SIZE,
                                                                      dim=IMAGE_DIMS,
                                                                      shuffle=SHUFFLE,
                                                                      ext="val")

model = ModelRepository(sys.argv[3], IMAGE_DIMS, len(file_names), BATCH_SIZE).get_model()

model.build([IMAGE_DIMS[0], IMAGE_DIMS[1], len(file_names)])
print(model.summary())
starttime = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

csv_logger = tf.keras.callbacks.CSVLogger("../logs/" + sys.argv[3] + "_" + starttime + ".csv", append=False,
                                          separator=",")

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, min_lr=0.0001)

log_dir = "../logs/tb_outputs/" + starttime
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit_generator(train_data_generator, validation_data=validation_data_generator, epochs=EPOCH,
                              shuffle=SHUFFLE, callbacks=[csv_logger, reduce_lr, tb_callback], steps_per_epoch=EPOCH_LIMIT,
                              validation_steps=EPOCH_LIMIT)

os.system("mv " + "../logs/" + sys.argv[3] + "_" + starttime + ".csv ../logs/" + sys.argv[3] + "_" + starttime
          + "_completed.csv ")

# TODO: Add log file writer for config of the run with same file name as csv log file.

model.save("../output_models/" + sys.argv[3] + "_" + starttime)
