from utils import raster_data_generator
from utils.train_test_splitter import TrainTestValidateSplitter
from utils.input_reader import InputReader
import sys
from models.model_repository import ModelRepository
import numpy as np
import tensorflow as tf

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

'''
if EPOCH_LIMIT is not None:
    test_epoch_limit = int(np.floor(EPOCH_LIMIT * VALIDATE_SIZE))
    validation_epoch_limit = int(np.floor(EPOCH_LIMIT * TEST_SIZE))
    train_list = train_list[:EPOCH_LIMIT]
    test_list = test_list[:EPOCH_LIMIT]
    validation_list = validation_list[:EPOCH_LIMIT]
'''

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

csv_logger = tf.keras.callbacks.CSVLogger("./csv_logs/" + sys.argv[3], append=False, separator=",")
history = model.fit_generator(train_data_generator, validation_data=validation_data_generator, epochs=EPOCH,
                              shuffle=SHUFFLE, callbacks=[csv_logger], steps_per_epoch=EPOCH_LIMIT,
                              validation_steps=EPOCH_LIMIT)

model.save("./output_models/" + sys.argv[3])

loss, acc = model.evaluate_generator(test_data_generator)
print("Test accuracy: {:5.2f}%".format(100 * acc))
model.save("./output_models/" + sys.argv[3])
