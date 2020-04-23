import os

input_dir = "/home/nagellette/Desktop/mass/mass_roads/valid/sat/sat_img/"
remove_dir = "/home/nagellette/Desktop/mass/mass_roads/valid/map/map_img/"

input_dir_files = os.listdir(input_dir)
remove_dir_files = os.listdir(remove_dir)

i = 0
for input_file in input_dir_files:
    if input_file in remove_dir_files:
        print(input_file + " in directory.")
    else:
        print("#     " + input_file + " not in directory.")
        # os.system("rm -rf " + input_dir + input_file)
        i += 1

print(i)
