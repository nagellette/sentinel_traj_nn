'''
Hard coded values for output filename extensions.
'''


def get_file_extension(file_name):
    if "_B01_" in file_name:
        return "_B01"
    elif "_B02_" in file_name:
        return "_B02"
    elif "_B03_" in file_name:
        return "_B03"
    elif "_B04_" in file_name:
        return "_B04"
    elif "_B05_" in file_name:
        return "_B05"
    elif "_B06_" in file_name:
        return "_B06"
    elif "_B07_" in file_name:
        return "_B07"
    elif "_B08_" in file_name:
        return "_B08"
    elif "_B8A_" in file_name:
        return "_B8A"
    elif "_B09_" in file_name:
        return "_B09"
    elif "_B11_" in file_name:
        return "_B11"
    elif "_B12_" in file_name:
        return "_B12"
    elif "_traj_count" in file_name:
        return "_traj_count"
    elif "_speed_max" in file_name:
        return "_speed_max"
    elif "_speed_avg" in file_name:
        return "_speed_avg"
    else:
        print("'{}' is not listed in utils.get_file_extension.py. Setting to unknown. Output can be erroneous.")
        return "_unknown"
