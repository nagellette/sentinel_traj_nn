import pandas as pd


def get_sample_list(sample_path, sample_file_name, sample_count, dataset_index):
    df = pd.read_csv(sample_path + sample_file_name, nrows=sample_count)
    df["index"] = dataset_index
    temp = df.values.astype('int64').tolist()

    return_list = []
    for i in temp:
        return_list.append([i, dataset_index])

    return return_list
