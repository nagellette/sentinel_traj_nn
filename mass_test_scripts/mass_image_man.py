import os
from matplotlib.image import imread, imsave
from tensorflow.keras.preprocessing.image import load_img, save_img
import tensorflow as tf

input_dir = "/Users/gengec/Desktop/code/data/mass_roads/train/map/map_img/"
output_dir = "/Users/gengec/Desktop/code/data/mass_roads/train/map_new/map_img/"
input_files = os.listdir(input_dir)

for file in input_files:
    image_file = tf.keras.preprocessing.image.img_to_array(load_img(input_dir + file, color_mode='grayscale'))
    print(image_file.shape)
    #image_file = image_file / 255.0
    #save_img(output_dir + file, image_file, data_format='channels_last')
