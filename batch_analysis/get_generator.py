from utils.raster_data_generator import RasterDataGenerator
from utils.input_reader import InputReader
from get_sample_list import get_sample_list
import sys
import ast


def get_generator(input_files,
                  test_list,
                  batch_size,
                  image_dims,
                  image_outputs):
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


samples = get_sample_list("./test_samples/", "_test_list_ist.csv", sample_count=1000,
                          dataset_index=0)

print(sys.argv[0])

a = get_generator(input_files=ast.literal_eval(sys.argv[1]), test_list=samples, batch_size=4, image_dims=(512, 512),
                  image_outputs="./")

print(a.__getitem__())
