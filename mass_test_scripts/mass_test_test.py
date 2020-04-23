import tensorflow as tf
from matplotlib.image import imsave

from utils.custom_generator import DataGenerator

IMAGE_DIMS = (256, 256)

model = tf.keras.models.load_model("./output_models/mass", compile=True)

valid_data = '/Users/gengec/Desktop/code/data/mass_roads/valid/sat/sat_img/'
valid_label = '/Users/gengec/Desktop/code/data/mass_roads/valid/map/map_img/'

validation_generator = DataGenerator(valid_data, valid_label, batch_size=1, dim=IMAGE_DIMS, n_channels=3, n_classes=2,
                                     shuffle=False)

predictions = model.predict_generator(validation_generator, steps=40)
print(predictions.shape)
print(len(predictions))

for i in range(len(predictions)):
    imsave("/Users/gengec/Desktop/code/sentinel_traj_nn/debug/images/mass/" + str(i) + "_0.png",
           predictions[i, :, :, 0])
    imsave("/Users/gengec/Desktop/code/sentinel_traj_nn/debug/images/mass/" + str(i) + "_1.png",
           predictions[i, :, :, 1])
