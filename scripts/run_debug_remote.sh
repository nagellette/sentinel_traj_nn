#!/bin/bash
model_name=unet
config_name=model_config_debug_remote.json
input_name=input_files_remote_small_msi.json

current_time=$(date "+%Y.%m.%d-%H.%M.%S")

new_fileName=runtime_log.$current_time

#SBATCH -p debug
#SBATCH -A ngengec
#SBATCH -J gpu-test-ngengec
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gengec@itu.edu.tr
#SBATCH --error=/truba/home/ngengec/error_logs/
#SBATCH --output=/truba/home/ngengec/output_logs/
#SBATCH --constraint=akya-cuda

python /truba/home/ngengec/sentinel_traj_nn/run_model.py \
  /truba/home/ngengec/sentinel_traj_nn/model_config_files/$config_name \
  /truba/home/ngengec/sentinel_traj_nn/model_config_files/$input_name $model_name >>/truba/home/ngengec/runtime_logs/$new_fileName
