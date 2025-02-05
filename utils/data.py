import numpy as np
import os.path as op
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
from utils.custom_transforms import Cutout
from utils.autoaugment import CIFAR10Policy, ImageNetPolicy

assert 0, "You should specify the path to your data/ folder"
data_path = 'enter path to data folder'

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        #transforms.ColorJitter(brightness=63/255)
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10(data_path, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10(data_path, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255),
        CIFAR10Policy(),
        transforms.ToTensor(),
        Cutout(n_holes=1,length=16)
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    ]

    class_order = np.arange(100).tolist()
    #icarl fixed class order :
    #class_order = [87,  0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18, 24, 32, 45, 88, 11, 4,
    # 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83,17, 81, 41, 12, 37, 59, 25, 20, 80, 73,1, 28,  6, 46, 62, 82, 53,  9,
    # 31, 75,38, 63, 33, 74, 27, 22, 36,  3, 16, 21,60, 19, 70, 90, 89, 43,  5, 42, 65, 76,40, 30, 23, 85,  2, 95, 56,
    # 48, 71, 64,98, 13, 99,  7, 34, 55, 54, 26, 35, 39]
    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100(data_path, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(data_path, train=False, download=True)
        self.classes = train_dataset.classes
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255),
        ImageNetPolicy(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        path = op.join(data_path, 'imagenet-object-localization-challenge')
        train_dir = op.join(path,'ILSVRC/Data/CLS-LOC/train/')
        test_dir = op.join(path,'ILSVRC/Data/CLS-LOC/val/')


        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.classes = train_dset.classes

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255),
        ImageNetPolicy(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        path = op.join(data_path, 'imagenet-object-localization-challenge')
        train_dir = op.join(path,'ILSVRC/Data/CLS-LOC/train/')
        test_dir = op.join(path,'ILSVRC/Data/CLS-LOC/val/')

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.classes = train_dset.classes
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)