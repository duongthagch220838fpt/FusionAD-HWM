# Install and configure CUDA 11.6
### 1. Install liburcu6t
```
sudo add-apt-repository ppa:cloudhan/liburcu6
sudo apt update
sudo apt install liburcu6
```
### 2. Install cuda
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get install -y cuda-toolkit-11-6
```
### 3. Configure CUDA Path
```
ls /usr/local | grep cuda
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-11.6 /usr/local/cuda
export PATH=/usr/local/cuda-11.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
```
### 4. Check CUDA version
```
nvcc --version
```
<br><br>

# Setup environment

### 1. Install miniconda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
### 2. Activate conda
```
bash Miniconda3-latest-Linux-x86_64.sh -b
source ~/miniconda3/bin/activate
```
### 3. Create and activate environment
```
conda create -n pytorch_env python=3.10
conda init pytorch_env
conda activate pytorch_env
```
### 4. Install requirements
Install pytorch version that works CUDA 11.6
```
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
```
Install pointnet2_ops and other libraries
```
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
cd FusionAD
pip install -r requirements.txt
```

<br><br>

# Install and preprocess the dataset

### 1. Install dataset
Install from source
```
wget "https://www.mydrive.ch/shares/45920/dd1eb345346df066c63b5c95676b961b/download/428824485-1643285832/mvtec_3d_anomaly_detection.tar.xz" -O mvtec_3d_anomaly_detection.tar.xz
```
Extract the file
```
tar -xvf mvtec_3d_anomaly_detection.tar.xz
```

### 2. Preprocess dataset
Install packages for preprocessing
```
pip install --ignore-installed plotly blinker Flask dash open3d
```
Run preprocessing for each class in the dataset with argument is the path of the class. Example (class "bagel"):
```
python preprocess_mvtec.py ./datasets/mvtec3d/bagel
```

<br><br>

# Training
Run training file with argument class_name for each class. Example (class "bagel"):
```
python training.py --class_name bagel
```
Then check the checkpoints folder if the weights are saved successfully

<br><br>

# Evaluation
Run the inference file with arguments --visualize_plot and --produce_qualitatives true, for each class. Example (class "bagel"):
```
python inference.py --visualize_plot --produce_qualitatives --class_name bagel
```
Then check the results folder if the quantitative and qualitative results are saved correctly.

After run inference for all classes, run the aggregate_results.py to generate the result table in LaTex format with the argument is the quantitative folder path
```
python aggregate_results.py ./results/quantitatives_mvtec
```