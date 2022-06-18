import tensorflow as tf


def get_fusion_layer(fusion_type, input1, input2):
    """
    Returning the preferred fusion type:

    :param fusion_type: Preferred fusion type
    :param input1: first input
    :param input2: second input

    """
    if fusion_type == "average":
        return tf.keras.layers.Average()([input1, input2])
    elif fusion_type == "sum":
        return tf.keras.layers.Add()([input1, input2])
    elif fusion_type == "multiply":
        return tf.keras.layers.Multiply()([input1, input2])
    elif fusion_type == "concat":
        return tf.keras.layers.Concatenate(axis=3)([input1, input2])
    else:
        print("Fusion type not defined!")
        return -1
