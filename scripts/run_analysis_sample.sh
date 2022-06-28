#!/bin/bash
#SBATCH -p akya-cuda
#SBATCH -A ngengec
#SBATCH -J gpu-test-ngengec
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gengec@itu.edu.tr
#SBATCH --error=/truba/home/ngengec/error_logs/err_%j
#SBATCH --output=/truba/home/ngengec/output_logs/out_%j
#SBATCH --time=0-23:59:59

home_path=/truba/home/ngengec/

python ${home_path}sentinel_traj_nn/run_analysis.py "163159" "msi" "ist"