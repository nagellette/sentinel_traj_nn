### Definition of runtime options

- **Model parameters:** Stored within ```./model/config_files/model_config_*``` files stores model parameters
- **Runtime scripts:** Stored under ```./scripts/```. Remote script files contains ```home_path``` for home directory, ```config_name```, ```model_name``` for chosen model(options are listed below)
    - ```unet```
    - ```srcnn-unet```
  
### Local Pycharm setup for run_model.Pycharm
- Add following to the run_model.py file "Parameters" section in run options dialog: ```"./model_config_files/model_config_debug_local.json" "./model_config_files/input_files_local_small_msi_only.json" "unet"```