# import the necessary packages
import torch
import os

# define the parent data dir followed by the training and test paths
BASE_PATH = "dataset"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
TEST_PATH = os.path.join(BASE_PATH, "val")
SAMPLE_PATH = os.path.join(BASE_PATH, "adinkra/symbols")

# specify ImageNet mean and standard deviation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# specify training hyperparameters
IMAGE_SIZE = 128
BATCH_SIZE = 32
PRED_BATCH_SIZE = 4
EPOCHS = 50
LR = 0.0001
MOM = 0.0
NUM_WORK = 4

# determine the device type 
DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# define paths to store training plot and trained model
PLOT_PATH = os.path.join("output_", "model_")
MODEL_PATH = os.path.join("output_", "model_")
FEAT_PATH = os.path.join("features", "")