from tensorflow.keras.metrics import Recall, Precision, BinaryAccuracy
from utils.custom_metrics import mean_iou, MeanIoU

'''
Control function for hard coded metric definitions.
'''


def get_metrics(batch_size):
    return ["accuracy",
            BinaryAccuracy(threshold=0.5),
            Precision(),
            Recall(),
            mean_iou,
            MeanIoU(num_classes=2, batch_size=batch_size, name="m_IoU")]
