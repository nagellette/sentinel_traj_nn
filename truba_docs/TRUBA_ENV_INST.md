### Truba Python Environment Installation
- Install Miniconda
```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

- Update conda and create new environment
```shell
eval "$(/truba/home/$USER/miniconda3/bin/conda shell.bash hook)"
conda update conda
conda create --name tensorflow_2.15
conda activate tensorflow_2.15
```

- Install Python and GDAL with conda (CUDA 12.2 / cuDNN 8.9 are installed by pip via `tensorflow[and-cuda]`, GPU driver must be >= 525, check with `nvidia-smi`)
```shell
conda install python=3.11
conda install -c conda-forge gdal=3.13.1 numpy=1.26.4
```

- Install `requirements.txt` with pip (comment out GDAL while installing to remote)
```shell
pip install -r requirements.txt
```

- Update `~/.bash_profile` with:
```shell
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/truba/home/ngengec/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/truba/home/ngengec/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/truba/home/ngengec/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/truba/home/ngengec/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate tensorflow_2.15
```

- Check file counts of folders
```shell
cd ~/
find . -type f | cut -d/ -f2 | sort | uniq -c
```