import tensorflow as tf

from models.model_repository import ModelRepository
from utils.custom_generator import DataGenerator

IMAGE_DIMS = (256, 256)

model = ModelRepository("unet", IMAGE_DIMS, 3, 1).get_model()

train_data = '/home/nagellette/Desktop/mass/mass_roads/train/sat/sat_img/'
train_label = '/home/nagellette/Desktop/mass/mass_roads/train/map/map_img/'

valid_data = '/home/nagellette/Desktop/mass/mass_roads/valid/sat/sat_img/'
valid_label = '/home/nagellette/Desktop/mass/mass_roads/valid/map/map_img/'

training_generator = DataGenerator(train_data, train_label, batch_size=1, dim=IMAGE_DIMS, n_channels=3, n_classes=2,
                                   shuffle=False)
validation_generator = DataGenerator(valid_data, valid_label, batch_size=1, dim=IMAGE_DIMS, n_channels=3, n_classes=2,
                                     shuffle=False)

model.build([256, 256, 3])
print(model.summary())

csv_logger = tf.keras.callbacks.CSVLogger("../output/logs/mass.csv", append=False, separator=",")
history = model.fit_generator(training_generator, epochs=40,
                              shuffle=True, callbacks=[csv_logger], steps_per_epoch=200, validation_steps=20,
                              validation_data=validation_generator
                              )

model.save("../output_models/mass")
