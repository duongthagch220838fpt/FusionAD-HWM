# Setting FusionAD
## Note: The setting is suitable only for devices with GPU - Check the GPU with this code below:
```sh 
nvidia-smi
```

## For OS:

### 1. Install miniconda
Detail document to install miniconda for OS: https://docs.anaconda.com/miniconda/miniconda-install/

### 2. Install requirements
Install pytorch version that works CUDA 11.8 for OS.
```
pip uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

You should install all the necessary dependencies in the `./requirements.txt` file.
```
pip install requirements.txt
```

### 3. Download data
#### Structure dataset
    # Folder structure is as follows:
    # data/classname
    # ├── annotation
    # │   ├── *.json
    # ├── anomaly
    # │   ├── low-light
    # │   │   ├── *.jpg
    # │   ├── well-light
    # │   │   ├── *.jpg
    # ├── normal
    # │   ├── low-light
    # │   │   ├── *.jpg
    # │   ├── well-light
    # │   │   ├── *.jpg
    # └── gt
    #     ├── *.png
**"Not done"**

Temporary Solution (only download the medicine dataset): 
```
python download.py
python rename.py data/medicine # Check and Edit structure dataset.
```

### 4. Run train.sh
To train the network refer to the example in `train.sh`

**Note**: The training script will log by wandb; you need to create an account from: https://wandb.ai/site When run, you have the option to use this; you can access wandb to generate an API in wandb and paste this when the connection requires the API.

The `train.py` script train the network.
```
python train.py --dataset_path data --class_name medicine
```

You can specify the following options:
   - `--dataset_path`: Path to the root directory of the dataset.
   - `--checkpoint_folder`: Path to the directory of the checkpoints, i.e., `checkpoints/`.
   - `--class_name`: Class on which the FADs are trained.
   - `--epochs_no`: Number of epochs for FADs optimization.
   - `--batch_size`: Number of samples per batch for FADs optimization.


### 5. Run infer.sh

The `infer.py` script test the trained model. It can be used to generate anomaly maps.

**batch_size in inference always 1.**
```
python infer.py --dataset_path data --class_name medicine
```
You can specify the following options:
   - `--dataset_path`: Path to the root directory of the dataset.
   - `--checkpoint_folder`: Path to the directory of the checkpoints (checkpoint from train), i.e., `checkpoints/`.
   - `--class_name`: Class on which the FADs was trained.
   - `--epochs_no`: Number of epochs used in FADs optimization.
   - `--batch_size_train`: Number of samples per batch employed when training.
   - `--qualitative_folder`: Folder on which the anomaly maps are saved.
   - `--quantitative_folder`: Folder on which the metrics are saved.
   - `--visualize_plot`: Flag to visualize qualitatived during inference.
   - `--produce_qualitatives`: Flag to save qualitatived during inference.


## Structure of Folder After Inference:
```
├───data
│   └───medicine
│       ├───annotation
│       ├───anomaly
│       │   ├───low-light
│       │   └───well-light
│       ├───gt
│       └───normal
│           ├───low-light
│           └───well-light
├───models
├───processing
├───results
│   ├───qualitatives_medicine
│   │   └───medicine_1_1
│   │       └───medicine
│   └───quantitatives_medicine
└───utils
```






