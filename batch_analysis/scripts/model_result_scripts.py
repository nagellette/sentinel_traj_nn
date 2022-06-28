import os
import sys
import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd

sys.path.insert(1, '../../../')

from utils.input_reader import InputReader
from utils.custom_losses import DiceLoss
from utils.custom_metrics import MeanIoU

"""
Scripts that are used in model output visualize notebook.
"""


# reading input path, report model type and file paths
def read_inputs_report(model_path):
    print("Working path: {}".format(model_path))

    root_file_list = os.listdir(model_path)
    model_type = "Not available"

    for file in root_file_list:
        if "completed" in file:
            model_type = file.split("_")[0]
            print("Model in use: {}".format(model_type))

    image_file_list = os.listdir("{}/images/".format(model_path))
    if len(image_file_list) > 0:
        print("# available test images: {}".format(len(image_file_list)))
    else:
        print("Model test outputs are not available.")

    return root_file_list, image_file_list, model_type


# read model config files
def read_model_inputs(model_path):

    model_path_content = os.listdir(model_path)
    # read configuration image files
    config = InputReader("{}config.json".format(model_path))

    image_inputs = []
    if "inputs.json" in model_path_content:
        image_inputs.append(InputReader("{}inputs.json".format(model_path)))
    else:
        inputs_list = list(filter(lambda x: "input_" in x, model_path_content))
        for i in range(len(inputs_list)):
            image_inputs.append(InputReader("{}input_{}.json".format(model_path, i)))

    model_config_output = []
    # get model parameters from config file
    BATCH_SIZE = config.get_batch_size()
    model_config_output.append(["Batch size: {}".format(BATCH_SIZE), "batch_size", BATCH_SIZE])
    IMAGE_DIMS = config.get_image_dim()
    model_config_output.append(["Patch size: {}".format(IMAGE_DIMS), "image_dims", IMAGE_DIMS])
    OPTIMIZER, L_RATE, DECAY, MOMENTUM, NESTEROV = config.get_optimizer_parameters()
    model_config_output.append(["Optimizer: {}".format(OPTIMIZER), "optimizer", OPTIMIZER])
    model_config_output.append(["Learning rate: {}".format(L_RATE), "l_rate", L_RATE])
    model_config_output.append(["Decay: {}".format(DECAY), "decay", DECAY])
    model_config_output.append(["Momentum: {}".format(MOMENTUM), "momentum", MOMENTUM])
    model_config_output.append(["Nesterov: {}".format(NESTEROV), "nesterov", NESTEROV])
    EPOCH = config.get_epoch_count()
    model_config_output.append(["# of epochs: {}".format(EPOCH), "epochs", EPOCH])
    TRAIN_SIZE, TEST_SIZE, VALIDATE_SIZE = config.get_data_split()
    model_config_output.append(
        ["Train, Test, Validation split: {}, {}, {}".format(TRAIN_SIZE, TEST_SIZE, VALIDATE_SIZE),
         "train_test_validation", [TRAIN_SIZE, TEST_SIZE, VALIDATE_SIZE]])
    AUGMENT = config.get_augmentation()
    model_config_output.append(["Rotation augmentation: {}".format(AUGMENT), "augment", AUGMENT])
    OVERLAP = config.get_overlap()
    model_config_output.append(["Patch overlap ratio: {}".format(OVERLAP), "overlap", OVERLAP])
    SHUFFLE = config.get_shuffle()
    model_config_output.append(["Shuffle outputs: {}".format(SHUFFLE), "shuffle", SHUFFLE])
    SEED = config.get_seed()
    model_config_output.append(["Seeding: {}".format(SEED), "seed", SEED])
    EPOCH_LIMIT = config.get_epoch_limit()
    model_config_output.append(["# training steps per epoch: {}".format(EPOCH_LIMIT), "epoch_limit", EPOCH_LIMIT])
    TEST_MODEL = config.get_test_model()
    model_config_output.append(["Model tested?: {}".format(TEST_MODEL), "test_model", TEST_MODEL])
    if TEST_MODEL:
        TEST_MODEL_LENGTH = config.get_test_model_count()
        model_config_output.append(
            ["Model test size: {}".format(TEST_MODEL_LENGTH), "test_model_lengths", TEST_MODEL_LENGTH])
    VALIDATION_MODEL_LENGTH = config.get_validation_model_count()
    model_config_output.append(["Model validation size: {}".format(VALIDATION_MODEL_LENGTH), "validation_model_lengths",
                                VALIDATION_MODEL_LENGTH])
    LOSS = config.get_loss()
    model_config_output.append(["Loss function: {}".format(LOSS), "loss", LOSS])

    print(
        "\n====================================================================\nModel "
        "configuration:\n====================================================================")
    for conf in model_config_output:
        print(conf[0])

    print(
        "====================================================================\nInput files and standardization "
        "method:\n====================================================================")

    file_names_all = []
    for image_input in image_inputs:
        work_directory, file_names, label_path, mask = image_input.read_image()
        file_names_all.append(file_names)
        for file_name in file_names:
            print("File: {}, Standardization: {}".format(file_name[0], file_name[1]))
        print("====================================================================")

    return model_config_output, file_names_all


# read tensorflow model
def read_model(model_path):
    root_file_list = os.listdir(model_path)

    model_available = False
    model_folder_name = ""
    model = None

    for file in root_file_list:
        if "model" in file:
            model_available = True
            model_folder_name = file

    if model_available:
        saved_model_path = "{}{}".format(model_path, model_folder_name)
        with custom_object_scope({"dice_loss_soft": dice_loss_soft, "dice_loss": dice_loss, "mean_iou": mean_iou,
                                  "MeanIoU": MeanIoU, "DiceLoss": DiceLoss}):
            model = tf.keras.models.load_model(saved_model_path, compile=False)
            trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
            non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])
            print("Total parameters: {}\nTrainable Parameters: {}\nNon-trainable parameters: {}".format(
                trainable_count + non_trainable_count, trainable_count, non_trainable_count))
    else:
        print("Model is not available. Returning None.")

    return model


# reading and returning the metrics outputs
def get_metrics_log(model_path):
    root_file_list = os.listdir(model_path)

    log_available = False
    log_file_name = ""
    log = None

    for file in root_file_list:
        if "completed" in file:
            log_available = True
            log_file_name = file

    if log_available:
        log = pd.read_csv("{}{}".format(model_path, log_file_name))
    else:
        print("Metric logs are not available.")

    return log


def get_timekeeper(model_path):
    time_log = pd.read_csv("{}timekeeper.csv".format(model_path))

    output_time_log = []

    train_iterations = time_log[(time_log["mode"] == "train") & (time_log["stage"] == "iteration")]
    temp = train_iterations["duration"]
    output_time_log.append(
        "Training iterations   - Mean: {:.2f} Min: {:.2f} Max: {:.2f} Count: {:.2f}".format(temp.mean(), temp.min(),
                                                                                            temp.max(), temp.count()))

    validation_iterations = time_log[(time_log["mode"] == "validate") & (time_log["stage"] == "iteration")]
    temp = validation_iterations["duration"]
    output_time_log.append(
        "Validation iterations - Mean: {:.2f} Min: {:.2f} Max: {:.2f} Count: {:.2f}".format(temp.mean(), temp.min(),
                                                                                            temp.max(), temp.count()))

    train_overall = time_log[(time_log["mode"] == "train") & (time_log["stage"] == "overall")]
    temp = train_overall["duration"]
    output_time_log.append(
        "Training epochs       - Mean: {:.2f} Min: {:.2f} Max: {:.2f} Count: {:.2f}".format(temp.mean(), temp.min(),
                                                                                            temp.max(), temp.count()))

    validation_overall = time_log[(time_log["mode"] == "validate") & (time_log["stage"] == "overall")]
    temp = validation_overall["duration"]
    output_time_log.append(
        "Validation epochs     - Mean: {:.2f} Min: {:.2f} Max: {:.2f} Count: {:.2f}".format(temp.mean(), temp.min(),
                                                                                            temp.max(), temp.count()))

    for i in output_time_log:
        print(i)

    return output_time_log
