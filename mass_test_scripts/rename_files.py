import os

input_dir = "/home/nagellette/Desktop/mass/mass_roads/valid/sat/sat_img/"
input_files = os.listdir(input_dir)

for file in input_files:
    os.system("mv " + input_dir + file + " " + input_dir + file.replace(".tiff", ".tif"))
