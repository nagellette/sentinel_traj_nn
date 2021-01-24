import json
import sys


class InputReader:
    def __init__(self, file_path):
        """
        Input images file reader
        :param file_path: Input json file path.
        """
        self.file_path = file_path

        try:
            f = open(self.file_path)
            self.json_data = json.load(f)
        except OSError as err:
            print("Cannot open image {0}. ({1})".format(self.file_path, err))
        except:
            print("Unexpected error:", sys.exc_info()[0])
        else:
            print("File in use: {0}".format(self.file_path))

    def read_image(self):
        """
        Image input files json parser.
        :return: image file names and work directory path
        """
        if self.json_data["file_type"] == "image":
            image_array = self.json_data["input_files"]
            working_path = self.json_data["working_path"]
            label = self.json_data["label_file"]
            return working_path, image_array, label
        else:
            print("Input file is not image file.")

    def get_config_input(self):

        if self.json_data["file_type"] != "config":
            print("Input file is not config file.")

    def get_batch_size(self):
        try:
            return self.json_data["batch_size"]
        except KeyError:
            print("Batch size is not defined, setting 1.")
            pass
        except:
            raise
        return 1

    def get_image_dim(self):
        try:
            dims = self.json_data["image_dim"]
            return (dims["x"], dims["y"])
        except KeyError:
            print("Image dimension is not provided, using (572, 572).")
            pass
        except:
            raise
        return (572, 572)

    def get_epoch_count(self):
        try:
            return self.json_data["epochs"]
        except KeyError:
            print("Number of epochs is not provided, using 5.")
            pass
        except:
            raise
        return 5

    def get_data_split(self):
        try:
            data_split = self.json_data["data_split"]
            return data_split["train"], data_split["test"], data_split["validate"]
        except KeyError:
            print("Data split is not provided, using - Train: %70, Test: %15, Validation: %15.")
            pass
        except:
            raise
        return 0.7, 0.15, 0.15

    def get_augmentation(self):
        try:
            return self.json_data["augment_deg"]
        except KeyError:
            print("Augmentation is not provided, using 0.")
            pass
        except:
            raise
        return 0

    def get_overlap(self):
        try:
            return self.json_data["overlap"]
        except KeyError:
            print("Overlap is not provided, using 0.")
            pass
        except:
            raise
        return 0

    def get_shuffle(self):
        try:
            if self.json_data["shuffle"] is True:
                return True
            elif self.json_data["shuffle"] is False:
                return False
        except KeyError:
            print("Shuffle is not provided, will not be shuffled.")
            pass
        except:
            raise
        return False

    def get_seed(self):
        try:
            return self.json_data["seed"]
        except KeyError:
            print("Seed is not provided, using 0.")
            pass
        except:
            raise
        return 0

    def get_epoch_limit(self):
        try:
            return self.json_data["epoch_limit"]
        except KeyError:
            pass

    def get_test_model(self):
        try:
            if self.json_data["test_model"] is True:
                return True
            elif self.json_data["test_model"] is False:
                return False
        except KeyError:
            print("Test model is not provided, will not be tested.")
            pass
        except:
            raise
        return False

    def get_srcnn_count(self):
        try:
            return self.json_data["srcnn_count"]
        except KeyError:
            pass

    def get_test_model_count(self):
        try:
            return self.json_data["test_model_count"]
        except KeyError:
            pass

    def get_validation_model_count(self):
        try:
            return self.json_data["validation_model_count"]
        except KeyError:
            pass

    def get_output_path(self):
        try:
            return self.json_data["output_path"]
        except KeyError:
            print("Output path is not defined.")
            pass
        except:
            raise
        return 0

    def get_optimizer_parameters(self):
        try:
            optimizer_parameters = self.json_data["optimizer_parameters"]
            return optimizer_parameters["optimizer"], \
                   optimizer_parameters["l_rate"], \
                   optimizer_parameters["decay"], \
                   optimizer_parameters["momentum"], \
                   optimizer_parameters["nesterov"]
        except KeyError:
            print("Optimizer parameters are not provided, using - SGD with learning rate: 0.001, decay: 1e-6, "
                  "momentum: 0.9, nesterov: True.")
            pass
        except:
            raise
        return "SGD", 0.001, 1e-6, 0.9, True

    def get_loss(self):
        try:
            return self.json_data["loss_function"]
        except KeyError:
            print("Loss function is not defined, setting 'dice'.")
            pass
        except:
            raise
        return "dice"
