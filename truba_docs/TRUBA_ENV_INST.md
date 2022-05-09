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
conda create --name tensorflow_2.5
conda activate tensorflow_2.5
```

- Install Tensorflow dependencies and GDAL with conda
```shell
conda install python=3.8
conda install -c conda-forge cudatoolkit=11.2
conda install -c conda-forge cudnn=8.1
conda install -c conda-forge gdal=3.3.2
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

conda activate tensorflow_2.5
```

- Check file counts of folders
```shell
cd ~/
find . -type f | cut -d/ -f2 | sort | uniq -c
```