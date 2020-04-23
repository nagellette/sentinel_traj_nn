#!/bin/bash
#SBATCH -p debug
#SBATCH -A ngengec
#SBATCH -J gpu-test-ngengec
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gengec@itu.edu.tr
#SBATCH --error=/truba/home/ngengec/sentinel_traj_nn/err
#SBATCH --output=/truba/home/ngengec/sentinel_traj_nn/out
#SBATCH --constraint=barbun-cuda
#SBATCH --time=0:05:00

python /truba/home/ngengec/sentinel_traj_nn/run_model.py model_config_debug_remote.json input_files_remote.json unet >> /truba/home/ngengec/sentinel_traj_nn/output
