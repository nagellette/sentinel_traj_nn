from tensorflow.keras.metrics import Recall, Precision, BinaryAccuracy
from utils.custom_metrics import MeanIoU

'''
Control function for hard coded metric definitions.
'''


def get_metrics(batch_size):
    return ["accuracy",
            Precision(),
            Recall()]
