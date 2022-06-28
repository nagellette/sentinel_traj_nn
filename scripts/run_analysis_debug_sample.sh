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
#SBATCH --error=/truba/home/ngengec/error_logs/err_%j
#SBATCH --output=/truba/home/ngengec/output_logs/out_%j
#SBATCH --constraint=akya-cuda
#SBATCH --time=0-00:14:00

home_path=/truba/home/ngengec/
slurm_job_id=$SLURM_JOB_ID

python ${home_path}sentinel_traj_nn/run_analysis.py "163159" "msi" "ist"  >> ${home_path}runtime_logs/${slurm_job_id}