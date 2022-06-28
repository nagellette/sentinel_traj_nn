#!/bin/bash
#SBATCH -p akya-cuda
#SBATCH -A ngengec
#SBATCH -J gpu-evaluation-ngengec
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gengec@itu.edu.tr
#SBATCH --error=/truba/home/ngengec/error_logs/err_%j
#SBATCH --output=/truba/home/ngengec/output_logs/out_%j
#SBATCH --time=0-23:59:59

model_id=$MODEL_ID
model_type=$MODEL_TYPE
model_area=$MODEL_AREA

slurm_job_id=$$SLURM_JOB_ID

home_path=/truba/home/ngengec/

python $${home_path}sentinel_traj_nn/run_analysis.py "{$$model_id}" "$${model_type}" "$${model_area}"  >> $${home_path}runtime_logs/$${$slurm_job_id}