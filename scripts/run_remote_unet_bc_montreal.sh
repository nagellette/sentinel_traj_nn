#!/bin/bash
#SBATCH -p akya-cuda
#SBATCH -A ngengec
#SBATCH -J ngengec-bc_unet
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gengec@itu.edu.tr
#SBATCH --error=/truba/home/ngengec/error_logs/err_%j
#SBATCH --output=/truba/home/ngengec/output_logs/out_%j
#SBATCH --time=0-15:00:00

model_name=unet
config_name=model_config_remote_bc.json
input_name=input_files_remote_small_msi.json

current_time=$(date "+%Y_%m_%d-%H_%M_%S")
slurm_job_id=$SLURM_JOB_ID

new_fileName=runtime_log_${current_time}_${slurm_job_id}
home_path=/truba/home/ngengec/

python ${home_path}sentinel_traj_nn/run_model.py \
  ${home_path}sentinel_traj_nn/model_config_files/${config_name} \
  ${home_path}sentinel_traj_nn/model_config_files/${input_name} ${model_name} ${current_time} >> ${home_path}runtime_logs/${new_fileName}

cp ${home_path}/output_logs/out_${slurm_job_id} ${home_path}/model_outputs/${slurm_job_id}_${model_name}_${current_time}/out_${model_name}_${current_time}
mv ${home_path}/output_logs/out_${slurm_job_id} ${home_path}/output_logs/out_${model_name}_${current_time}_${slurm_job_id}

cp ${home_path}/runtime_logs/${new_fileName} ${home_path}/model_outputs/${slurm_job_id}_${model_name}_${current_time}/runtime_log_${model_name}_${current_time}

cp ${home_path}/error_logs/err_${slurm_job_id} ${home_path}/model_outputs/${slurm_job_id}_${model_name}_${current_time}/err_${model_name}_${current_time}
mv ${home_path}/error_logs/err_${slurm_job_id} ${home_path}/error_logs/err_${model_name}_${current_time}_${slurm_job_id}
