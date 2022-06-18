### Definition of runtime options

- **Model parameters:** Stored within ```./model/config_files/model_config_*``` files stores model parameters
- **Runtime scripts:** Stored under ```./scripts/```. Remote script files contains ```home_path``` for home directory, ```config_name```, ```model_name``` for chosen model(options are listed below)
  - ```test_model``` (deprecated)
  - ```unet```
  - ```unetlight```
  - ```srcnn_unet``` (not fully functional, yet)
  - ```resunet```
  - ```resunetlight```
  - ```dlinknet```
  - ```unet_traj_type1```: Unet with late fusion with no model in trajectory stream.
  - ```unet_traj_type2```: Unet with late fusion with Unet model in trajectory stream.
  
### Local Pycharm setup for run_model.Pycharm
- Add following to the run_model.py file "Parameters" section in run options dialog: ```"./model_config_files/model_config_debug_local.json" "./model_config_files/input_files_local_small_msi_only.json" "unet"```
- Second argument can be a list of input files for the multi area experiments: ```"./model_config_files/model_config_debug_local.json" "['./model_config_files/input_files_local_small_msi.json', './model_config_files/input_files_local_ist_msi.json']" "unet" "test"```
- Pass last argument as ```test``` for local experiments.

### run_experiment.py runtime args
- ```python run_experiments.py ./experiments/ist_montreal_all.csv ./scripts/run_remote_template.sh```

Experiment file template:

JOBDEVICE| JOBNAME          | MODELNAME  | MODELCONFIG                       | MODELINPUTS                                                                                 
---|------------------|------------|-----------------------------------|---------------------------------------------------------------------------------------------|
truba cluster name| model description| model name | model config file to be used      | model input file path list. multiple paths work if there is more than one work area.        |
akya-cuda| ngengec_resunet_bc_ist | resunet    | model_config_remote_bc.json       | ['/truba/home/ngengec/sentinel_traj_nn/model_config_files/input_files_remote_ist_msi.json'] 
akya-cuda| ngengec_dlinknet_bc_ist | dlinknet   | model_config_remote_bc_dlink.json | ['/truba/home/ngengec/sentinel_traj_nn/model_config_files/input_files_remote_ist_msijson']  
akya-cuda| ngengec_unet_bc_ist | unet       | model_config_remote_bc.json       | ['/truba/home/ngengec/sentinel_traj_nn/model_config_files/input_files_remote_ist_msi.json'] 

* Seperator "|". refer to experiment table for templates.