# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 09:42:52 2021

@author: Micky

The *config.py* file in the search folder stores information such as
parameters, initial settings, configurations for our code.
"""


# import the necessary packages
import torch
import os


# define the parent data dir followed by the training and test paths
# BASE_PATH = "dataset"
# TRAIN_PATH = os.path.join(BASE_PATH, "training_set")
# TEST_PATH = os.path.join(BASE_PATH, "test_set")


# specify path to the flowers and mnist dataset
#ADINKRA_DATASET_PATH = "dataset/adinkra/dataset"
TEST_IMG_PATH = "dataset/adinkra/symbols"
TRAIN_DATASET_PATH = "dataset/train"
TEST_DATASET_PATH = "dataset/val"
#ADINKRA_DATASET_PATH = "../Adinkra_Symbols"

# specify the paths to our training and validation set
#TRAIN = "dataset/train"
#VAL = "dataset/val"


# initialize the initial learning rate, batch size, and number of
# epochs to train for


# specify ImageNet mean and standard deviation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# set the input height and width
INPUT_HEIGHT = 128
INPUT_WIDTH = 128

# set the batch size and validation data split
INIT_LR = 1e-4
BS = 128
EPOCHS = 50

#VAL_SPLIT = 0.1

# define the number of devices used for training
NUM_WORKERS = 4

# determine the device type 
DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"

SAVE = "output"
MODEL_PATH = "output/pth/"

#augmentation parameter
#augment = 1000
