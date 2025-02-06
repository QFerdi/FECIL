# Introduction
This is the official pytorch implementation of the FECIL method for class incremental learning. This impementation is based on the PyCIL toolbox [1].

The FECIL method trains incrementally a CNN model by first expanding its feature extractor to adapt to new classes and then leverages the Rehearsal-Cutmix augmentation to reduce catastrophic forgetting during the compression step that compresses the model back to its original size.

# Requirements

- pytorch 1.8.1
- torchvision 0.6.0
- tensorboard
- tqdm
- pyyaml
- numpy
- scipy
- scikit-learn
- matplotlib

# How To use

Once the dependencies are installed, a few steps are required to run this code.

### Clone
clone this GitHub repository : 
```sh
git clone https://github.com/QFerdi/FECIL.git
```
### Set the datasets path
In utils/data.py, set the variable data_path to your local data folder. The code will automatically download the CIFAR100 dataset in this folder if it is not already in it but the ImageNet-1000 dataset should be downloaded separately if you want to train on ImageNet-1000 or ImageNet-100.

the following lines in utils/data.py should be modified (the assertion should be removed and the path should lead to your data folder) : 
```python
assert 0, "You should specify the path to your data/ folder"
data_path = 'enter path to data folder'
```

### Run the desired experiment
The config files used for experiments using the ResNet32 or ResNet18 architectures are located respectively in exps/Resnet32 and exps/Resnet18. The config files in these folders correspond to the different experimental protocols used in the paper. For example, fecil_b0_5.yaml correspond to the b0 protocol with 5 classes added during each incremental step.

In order to train FECIL with the choosen configuration, simply run the following command : 
```sh
python main.py --config exps/Resnet18/fecil_b0_10.yaml
```
### Citations

    [1] D.-W. Zhou, F.-Y. Wang, H.-J. Ye, D.-C. Zhan, Pycil: A python toolbox for class-incremental learning, 2023.