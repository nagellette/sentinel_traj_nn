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
  
### Local Pycharm setup for run_model.Pycharm
- Add following to the run_model.py file "Parameters" section in run options dialog: ```"./model_config_files/model_config_debug_local.json" "./model_config_files/input_files_local_small_msi_only.json" "unet"```
- Second argument can be a list of input files for the multi area experiments: ```"./model_config_files/model_config_debug_local.json" "['./model_config_files/input_files_local_small_msi.json', './model_config_files/input_files_local_ist_msi.json']" "unet" "test"```
- Pass last argument as ```test``` for local experiments.

### run_experiment.py runtime args
- ```python run_experiments.py ./experiments/ist_montreal_all.csv ./scripts/run_remote_template.sh```