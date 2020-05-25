from tensorflow.keras.callbacks import Callback
from datetime import datetime
from csv import writer


class TimeKeeper(Callback):
    def __init__(self, log_path):
        '''
        Callback class for time spent logger for training, validation and test runs by epoch and iteration
        :param log_path: output folder for timekeeper.csv file
        '''
        #output file
        self.log_path = log_path + "timekeeper.csv"

        # counters
        self.epoch_counter = 0
        self.batch_counter = 0
        self.validate_counter = 0
        self.predictions_counter = 0

        # training start and end time keepers
        self.train_start = 0
        self.train_end = 0

        # training epoch start and end time keepers
        self.epoch_start = 0
        self.epoch_end = 0

        # start and end time keepers for one batch of training iteration
        self.batch_start = 0
        self.batch_end = 0

        # training epoch validation start and end time keepers
        self.validate_start = 0
        self.validate_end = 0

        # start and end time keepers for one batch of validation iteration
        self.validate_batch_start = 0
        self.validate_batch_end = 0

        # predictions start and end time keepers
        self.predictions_start = 0
        self.predictions_end = 0

        # start and end time keepers of each prediction iteration
        self.prediction_start = 0
        self.prediction_end = 0

    # csv file writing method
    def append_list_as_row(self, file_name, list_of_elem):
        # Open file in append mode
        with open(file_name, 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            # Add contents of list as last row in the csv file
            csv_writer.writerow(list_of_elem)

    def on_train_begin(self, logs=None):
        self.train_start = datetime.now()
        print("Training started at {}".format(self.train_start))
        file_header = ["mode", "stage", "epoch", "iteration", "val_iteration", "duration"]
        self.append_list_as_row(self.log_path, file_header)

    def on_train_end(self, logs=None):
        self.train_end = datetime.now()
        training_duration = (self.train_end - self.train_start).total_seconds()
        print("Training took: {}".format(str(training_duration)))
        write_output = ["train", "train_end", self.epoch_counter, self.epoch_counter, "-", training_duration]
        self.append_list_as_row(self.log_path, write_output)

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start = datetime.now()

    def on_train_batch_end(self, batch, logs=None):
        self.batch_end = datetime.now()
        batch_duration = (self.batch_end - self.batch_start).total_seconds()
        write_output = ["train", "iteration", self.epoch_counter, self.batch_counter, "-", batch_duration]
        self.append_list_as_row(self.log_path, write_output)
        self.batch_counter += 1

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = datetime.now()
        self.batch_counter = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_end = datetime.now()
        epoch_duration = (self.epoch_end - self.epoch_start).total_seconds()
        write_output = ["train", "overall", self.epoch_counter, self.batch_counter, "-", epoch_duration]
        self.append_list_as_row(self.log_path, write_output)
        self.epoch_counter += 1

    def on_test_batch_begin(self, batch, logs=None):
        self.validate_batch_start = datetime.now()
        self.validate_counter = 0

    def on_test_batch_end(self, batch, logs=None):
        self.validate_batch_end = datetime.now()
        validate_batch_duration = (self.validate_batch_end - self.validate_batch_start).total_seconds()
        write_output = ["validate", "iteration", self.epoch_counter, "-", self.validate_counter, validate_batch_duration]
        self.append_list_as_row(self.log_path, write_output)
        self.validate_counter += 1

    def on_test_begin(self, logs=None):
        self.validate_start = datetime.now()

    def on_test_end(self, logs=None):
        self.validate_end = datetime.now()
        validate_duration = (self.validate_end - self.validate_start).total_seconds()
        write_output = ["validate", "overall", self.epoch_counter, "-", self.validate_counter, validate_duration]
        self.append_list_as_row(self.log_path, write_output)

    def on_predict_begin(self, logs=None):
        self.predictions_start = datetime.now()

    def on_predict_end(self, logs=None):
        self.predictions_end = datetime.now()
        predictions_duration = (self.predictions_end - self.predictions_start).total_seconds()
        write_output = ["predict", "overall", "-", "-", self.predictions_counter, predictions_duration]
        self.append_list_as_row(self.log_path, write_output)
        self.predictions_counter = 0

    def on_predict_batch_begin(self, batch, logs=None):
        self.prediction_start = datetime.now()

    def on_predict_batch_end(self, batch, logs=None):
        self.prediction_end = datetime.now()
        prediction_duration = (self.prediction_end - self.prediction_start).total_seconds()
        write_output = ["predict", "iteration", "-", "-", self.predictions_counter, prediction_duration]
        self.append_list_as_row(self.log_path, write_output)
        self.predictions_counter += 1
