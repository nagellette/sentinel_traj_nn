from utils.raster_data_generator import RasterDataGenerator
from utils.input_reader import InputReader
from batch_analysis.get_sample_list import get_sample_list
import sys
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
import pandas as pd
import matplotlib.pyplot as plt
from utils.custom_metrics import create_boundary

from keras_unet_collection.transformer_layers import patch_extract, patch_embedding, SwinTransformerBlock, \
    patch_merging, patch_expanding
from keras_unet_collection.activations import GELU

# parameters
DIMS = (512, 512)
BATCH_SIZE = 4  # update this line depending on the input batch size
STEPS = 250  # proportionally update this line depending on the input batch size

# set input parameters
model_no = sys.argv[1]
model_type = sys.argv[2]
model_area = sys.argv[3]
output_folder = f"/truba/home/ngengec/evaluations/{model_no}_{model_type}/{model_area}/"
model_path = f"/truba/home/ngengec/evaluations/{model_no}_{model_type}/{model_no}.h5"
batch_path = "/truba/home/ngengec/sentinel_traj_nn/batch_analysis"

# read stats output file
result_file_all = pd.read_csv("{}/output_all.csv".format(batch_path))
result_file_removed = pd.read_csv("{}/output_removed.csv".format(batch_path))

# UPDATE these depending on the input dataset in use.
mont_msi = "input_files_remote_small_msi"  # input_files_remote_small_msi_rgb
mont_traj = "input_files_remote_small_msi_traj"  # input_files_remote_small_msi_rgb_traj
ist_msi = "input_files_remote_ist_msi"  # input_files_remote_ist_msi_rgb
ist_traj = "input_files_remote_ist_msi_traj"  # input_files_remote_ist_msi_rgb_traj

# set inputs
if model_area == "ist":
    samples = get_sample_list("{}/test_samples/".format(batch_path), "_test_list_ist.csv", sample_count=1000,
                              dataset_index=0)
    if model_type == "traj":
        input_files = [f'/truba/home/ngengec/sentinel_traj_nn/model_config_files/{ist_traj}.json']
    else:
        input_files = [f'/truba/home/ngengec/sentinel_traj_nn/model_config_files/{ist_msi}.json']
elif model_area == "mont":
    samples = get_sample_list("{}/test_samples/".format(batch_path), "_test_list_mont.csv", sample_count=1000,
                              dataset_index=0)
    if model_type == "traj":
        input_files = [f'/truba/home/ngengec/sentinel_traj_nn/model_config_files/{mont_traj}.json']
    else:
        input_files = [f'/truba/home/ngengec/sentinel_traj_nn/model_config_files/{mont_msi}.json']
else:
    samples = get_sample_list("{}/test_samples/".format(batch_path), "_test_list_mont.csv", sample_count=500,
                              dataset_index=1)
    samples_ist = get_sample_list("{}/test_samples/".format(batch_path), "_test_list_ist.csv", sample_count=500,
                                  dataset_index=0)
    for sample in samples_ist:
        samples.append(sample)

    if model_type == "traj":
        input_files = [f'/truba/home/ngengec/sentinel_traj_nn/model_config_files/{ist_traj}.json',
                       f'/truba/home/ngengec/sentinel_traj_nn/model_config_files/{mont_traj}.json']
    else:
        input_files = [f'/truba/home/ngengec/sentinel_traj_nn/model_config_files/{ist_msi}.json',
                       f'/truba/home/ngengec/sentinel_traj_nn/model_config_files/{mont_msi}.json']


# generator
def get_generator(input_files,
                  test_list,
                  batch_size,
                  image_dims,
                  image_outputs=None):
    # read input file definitions
    image_inputs = []

    for input_file in input_files:
        image_inputs.append(InputReader(input_file))

    # set file paths from input file
    images = []
    for image_input in image_inputs:
        images.append(image_input.read_image())

    test_data_generator = RasterDataGenerator(inputs=images,
                                              generation_list=test_list,
                                              batch_size=batch_size,
                                              dim=image_dims,
                                              shuffle=False,
                                              ext="test",
                                              save_image_file=image_outputs,
                                              srcnn_count=0,
                                              non_srcnn_count=False)

    return test_data_generator


# create test generator
test_generator = get_generator(input_files=input_files, test_list=samples, batch_size=BATCH_SIZE, image_dims=DIMS)

# read model
model = tf.keras.models.load_model(model_path, custom_objects={"GELU": GELU,
                                                               "patch_extract": patch_extract,
                                                               "patch_embedding": patch_embedding,
                                                               "SwinTransformerBlock": SwinTransformerBlock,
                                                               "patch_merging": patch_merging,
                                                               "patch_expanding": patch_expanding
                                                               })

# get predictions
predictions = model.predict(test_generator, steps=STEPS)

# create new test generator to get images and labels manually to write on disk
test_generator = get_generator(input_files=input_files, test_list=samples, batch_size=BATCH_SIZE, image_dims=DIMS)

# set iterator constants:
# prediction counter
prediction_count = 0

# create measure output list size
list_size = (STEPS * BATCH_SIZE)

# create measure output list for measures from tensorflow library
iou_tf = np.empty([list_size])
accuracy_tf = np.empty([list_size])
precision_tf = np.empty([list_size])
recall_tf = np.empty([list_size])
boundary_iou_tf = np.empty([list_size])

# create measure output list for measures from tensorflow-addons library
f1_tfa = np.empty([list_size])

# create stat output list
stat_output_list = []

# iterate over predictions and batches
for i in range(STEPS):

    # get next batch
    x_batch, y_batch = test_generator.__getitem__(i)

    # iterate withing batch to do the needed calculations and save the outputs
    for j in range(BATCH_SIZE):

        # create measure objects:
        # Measures from tensorflow library
        measure_iou_tf = tf.keras.metrics.MeanIoU(num_classes=2)
        measure_accuracy_tf = tf.keras.metrics.Accuracy()
        measure_precision_tf = tf.keras.metrics.Precision(thresholds=0.5)
        measure_recall_tf = tf.keras.metrics.Recall(thresholds=0.5)
        measure_boundary_iou_tf = tf.keras.metrics.MeanIoU(num_classes=2)

        # Measures from tensorflow-addons library
        measure_f1_tfa = tfa.metrics.F1Score(num_classes=2, threshold=0.5, average='micro')

        # read prediction probabilities and convert to binary prediction image in int type
        prediction_prob = predictions[prediction_count, :, :, 0]
        prediction = prediction_prob > 0.5
        prediction = prediction.astype(int)

        # save predictions as image
        plt.imshow(prediction_prob, cmap="gray")
        plt.imsave("{}{}_prob.jpeg".format(output_folder, str(prediction_count)), prediction_prob, cmap="gray")
        plt.imshow(prediction, cmap="gray")
        plt.imsave("{}{}_pred.jpeg".format(output_folder, str(prediction_count)), prediction, cmap="gray")

        # read label and convert to int type
        y = y_batch[j, :, :, 0]
        y = y.astype(int)

        # save label image
        plt.imshow(y, cmap="gray")
        plt.imsave("{}{}_label.jpeg".format(output_folder, str(prediction_count)), y, cmap="gray")

        y_boundary = create_boundary(y)
        prediction_boundary = create_boundary(prediction)

        # run measures and add to output list:
        # Measures from tensorflow library
        measure_iou_tf.update_state(y, prediction)
        measure_accuracy_tf.update_state(y, prediction)
        measure_precision_tf.update_state(y, prediction)
        measure_recall_tf.update_state(y, prediction)
        measure_boundary_iou_tf.update_state(y_boundary, prediction_boundary)

        # Measures from tensorflow-addons library
        measure_f1_tfa.update_state(y, prediction)

        iou_tf[prediction_count] = measure_iou_tf.result().numpy()
        accuracy_tf[prediction_count] = measure_accuracy_tf.result().numpy()
        precision_tf[prediction_count] = measure_precision_tf.result().numpy()
        recall_tf[prediction_count] = measure_recall_tf.result().numpy()
        boundary_iou_tf[prediction_count] = measure_boundary_iou_tf.result().numpy()

        f1_tfa[prediction_count] = measure_f1_tfa.result().numpy()

        # add stats to output list
        stat_output_list.append([prediction_count,
                                 measure_iou_tf.result().numpy(),
                                 measure_accuracy_tf.result().numpy(),
                                 measure_precision_tf.result().numpy(),
                                 measure_recall_tf.result().numpy(),
                                 measure_f1_tfa.result().numpy(),
                                 measure_boundary_iou_tf.result().numpy()])

        x = x_batch[j, :, :, :3]
        # save msi image
        plt.imshow(x, cmap="magma")
        plt.imsave("{}{}_msi.jpeg".format(output_folder, str(prediction_count)), x, cmap="magma")

        if x_batch.shape[3] == 5:
            x_traj = x_batch[j, :, :, 4]
            plt.imshow(x_traj, cmap="magma")
            plt.imsave("{}{}_traj.jpeg".format(output_folder, str(prediction_count)), x_traj, cmap="gray")

        prediction_count += 1

# save output stats
df_stat_output_list = pd.DataFrame(stat_output_list, columns=["sample_count",
                                                              "iou",
                                                              "accuracy",
                                                              "precision",
                                                              "recall",
                                                              "f1",
                                                              "boundary_iou"])
df_stat_output_list.to_csv("{}output_stats.csv".format(output_folder))

# remove fully overlapping (=empty) samples
remove = np.where(iou_tf == 1.)

# create removed stat output file
print("Mean IoU : {}/{}".format(iou_tf.mean(), iou_tf.std()))
print("Accuracy : {}/{}".format(accuracy_tf.mean(), accuracy_tf.std()))
print("Precision : {}/{}".format(precision_tf.mean(), precision_tf.std()))
print("Recall : {}/{}".format(recall_tf.mean(), recall_tf.std()))
print("Boundary IoU : {}/{}".format(boundary_iou_tf.mean(), boundary_iou_tf.std()))
print("F1 Score : {}/{}".format(f1_tfa.mean(), f1_tfa.std()))

df_data = [[model_path[-9:].replace(".h5", ""), model_type, model_area,
            iou_tf.mean(), iou_tf.std(),
            accuracy_tf.mean(), accuracy_tf.std(),
            precision_tf.mean(), precision_tf.std(),
            recall_tf.mean(), recall_tf.std(),
            f1_tfa.mean(), f1_tfa.std(),
            boundary_iou_tf.mean(), boundary_iou_tf.std()]]

df_temp = pd.DataFrame(df_data, columns=["model_no", "model_type", "model_area",
                                         "iou", "iou_std",
                                         "accuracy", "accuracy_std",
                                         "precision", "precision_std",
                                         "recall", "recall_std",
                                         "f1", "f1_std",
                                         "boundary_iou", "boundary_iou_std"])

# result_file_all = pd.concat([result_file_all, df_temp], axis=0)
df_temp.to_csv("{}/output_all.csv".format(batch_path), mode='a', header=False, index=False)

iou_tf = np.delete(iou_tf, remove)
accuracy_tf = np.delete(accuracy_tf, remove)
precision_tf = np.delete(precision_tf, remove)
recall_tf = np.delete(recall_tf, remove)
boundary_iou_tf = np.delete(boundary_iou_tf, remove)
f1_tfa = np.delete(f1_tfa, remove)

# create removed stat output file
print("Mean IoU : {}/{}".format(iou_tf.mean(), iou_tf.std()))
print("Accuracy : {}/{}".format(accuracy_tf.mean(), accuracy_tf.std()))
print("Precision : {}/{}".format(precision_tf.mean(), precision_tf.std()))
print("Recall : {}/{}".format(recall_tf.mean(), recall_tf.std()))
print("Boundary IoU : {}/{}".format(boundary_iou_tf.mean(), boundary_iou_tf.std()))
print("F1 Score : {}/{}".format(f1_tfa.mean(), f1_tfa.std()))

df_data = [[model_path[-9:].replace(".h5", ""), model_type, model_area,
            iou_tf.mean(), iou_tf.std(),
            accuracy_tf.mean(), accuracy_tf.std(),
            precision_tf.mean(), precision_tf.std(),
            recall_tf.mean(), recall_tf.std(),
            f1_tfa.mean(), f1_tfa.std(),
            boundary_iou_tf.mean(), boundary_iou_tf.std()]]

df_temp = pd.DataFrame(df_data, columns=["model_no", "model_type", "model_area",
                                         "iou", "iou_std",
                                         "accuracy", "accuracy_std",
                                         "precision", "precision_std",
                                         "recall", "recall_std",
                                         "f1", "f1_std",
                                         "boundary_iou", "boundary_iou_std"])

# result_file_removed = pd.concat([result_file_removed, df_temp], axis=0)
df_temp.to_csv("{}/output_removed.csv".format(batch_path), mode='a', header=False, index=False)
