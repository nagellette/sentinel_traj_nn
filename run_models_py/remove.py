import tensorflow as tf
import numpy as np

image1 = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
image2 = [[3, 4, 5], [3, 4, 5], [3, 4, 5]]

band_of_images_append1 = []
band_of_images_append1.append(image1)
band_of_images_append1.append(image2)

print(band_of_images_append1)

image3 = [[5, 6, 7], [5, 6, 7], [5, 6, 7]]
image4 = [[7, 8, 9], [7, 8, 9], [7, 8, 9]]

band_of_images_append2 = []
band_of_images_append2.append(image3)
band_of_images_append2.append(image4)

print(band_of_images_append2)

band_of_images_append1 = np.array(band_of_images_append1)
band_of_images_append2 = np.array(band_of_images_append2)

print(band_of_images_append1.shape)
print(band_of_images_append2.shape)

batch_of_images = np.stack([band_of_images_append1, band_of_images_append2], axis=0)

print(batch_of_images.shape)

image1 = np.array(image1)
input_shape = image1.shape
output_shape = input_shape + (2,)

print(input_shape)
print(output_shape)