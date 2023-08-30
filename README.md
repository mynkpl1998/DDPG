# Installation 

## Installing Mujoco

### Getting Binaries
```
mkdir ~/.mujoco/mujoco210/
wget https://github.com/deepmind/mujoco/releases/download/2.3.7/mujoco-2.3.7-linux-x86_64.tar.gz
tar -xvf mujoco-2.3.7-linux-x86_64.tar.gz
```

### Edit the .bashrc with the following path
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/mujoco-2.3.7/bin/
source ~/.bashrc
```

## Installing Python and required dependencies
```
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh
bash Miniconda3-py310_23.5.2-0-Linux-x86_64.sh
conda env create -f environment.yml
conda activate rl_envpy2
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

