from utils import raster_data_generator

train_data_generator = raster_data_generator.RasterDataGenerator(inputs=images,
                                                                 generation_list=train_list,
                                                                 batch_size=BATCH_SIZE,
                                                                 dim=IMAGE_DIMS,
                                                                 shuffle=SHUFFLE,
                                                                 ext="train",
                                                                 srcnn_count=SRCNN_COUNT,
                                                                 non_srcnn_count=False)