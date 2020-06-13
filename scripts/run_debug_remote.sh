#!/bin/bash
#SBATCH -p debug
#SBATCH -A ngengec
#SBATCH -J gpu-test-ngengec
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gengec@itu.edu.tr
#SBATCH --error=/truba/home/ngengec/error_logs/err
#SBATCH --output=/truba/home/ngengec/output_logs/out
#SBATCH --constraint=akya-cuda
#SBATCH --time=0-00:14:00

model_name=unet
config_name=model_config_debug_remote.json
input_name=input_files_remote_small_msi.json

current_time=$(date "+%Y.%m.%d-%H.%M.%S")

new_fileName=runtime_log.$current_time
home_path=/truba/home/ngengec/

python ${home_path}sentinel_traj_nn/run_model.py \
  ${home_path}sentinel_traj_nn/model_config_files/${config_name} \
  ${home_path}sentinel_traj_nn/model_config_files/${input_name} ${model_name} >>${home_path}runtime_logs/${new_fileName}

mv ${home_path}/error_logs/err ${home_path}/error_logs/err_${model_name}_${current_time}

mv ${home_path}/output_logs/out ${home_path}/output_logs/out_${model_name}_${current_time}
