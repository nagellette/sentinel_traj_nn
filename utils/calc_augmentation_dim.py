import math


def calc_augmentation_dim(length, angle):
    angle = math.radians(angle % 90.0)
    new_dim = int(length * (math.sin(angle) + math.cos(angle)))

    shift = int((new_dim - length) / 2)

    return new_dim, shift
