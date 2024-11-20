# USAGE
# python predict.py --model vgg
# python predict.py --model resnet


# import the necessary packages
from assets import _config
from assets.classifier import Classifier
from assets.datapipeline import get_dataloader 
import translate
from deep_cnn import DeeperCNN

from colorama import Fore, Style
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from torchvision.transforms import Normalize
from torchvision import transforms
from torch.nn import Softmax
from torch import nn
import matplotlib.pyplot as plt
import argparse
import torch
from tqdm import tqdm

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="vgg",
	choices=["cnn", "vgg", "resnet"], help="name of the backbone model")

args = vars(ap.parse_args())

# initialize test transform pipeline
imgTransform = Compose([
	Resize((_config.IMAGE_SIZE, _config.IMAGE_SIZE)),
	ToTensor(),
	Normalize(mean=_config.MEAN, std=_config.STD)
])

# calculate the inverse mean and standard deviation
invMean = [-m/s for (m, s) in zip(_config.MEAN, _config.STD)]
invStd = [1/s for s in _config.STD]


# define our denormalization transform
deNormalize = transforms.Normalize(mean=invMean, std=invStd)

# create the test dataset
testDataset = ImageFolder(_config.SAMPLE_PATH, imgTransform)

# initialize the test data loader
testLoader = get_dataloader(testDataset, _config.PRED_BATCH_SIZE)


# check if the name of the backbone model is CNN
if args["model"] == "cnn":
	# load our CNN model
	baseModel = DeeperCNN(num_classes=62)

# or, the backbone model we will be using is a VGG
elif args["model"] == "vgg":
	# load VGG-11 model
	baseModel = torch.hub.load("pytorch/vision:v0.10.0", "vgg11",
		pretrained=True, skip_validation=True)

# otherwise, the backbone model we will be using is a ResNet
elif args["model"] == "resnet":
	# load ResNet 18 model
	baseModel = torch.hub.load("pytorch/vision:v0.10.0", "resnet18",
		pretrained=True, skip_validation=True)

# build the custom model
model = Classifier(baseModel=baseModel.to(_config.DEVICE), numClasses=62, model=False)
model = model.to(_config.DEVICE)

# load the model state and initialize the loss function
model.load_state_dict(torch.load(_config.MODEL_PATH+args["model"]+'-'+str(_config.EPOCHS)+'.pth'))
lossFunc = nn.CrossEntropyLoss()
lossFunc.to(_config.DEVICE)

# initialize test data loss
testCorrect = 0
totalTestLoss = 0
soft = Softmax()
# moutput = nn.Softmax(dim=1)(moutput)[0]*100


# switch off autograd
with torch.no_grad():
	# set the model in evaluation mode
	model.eval()
	
	# loop over the validation set
	for (image, target) in tqdm(testLoader):
		
		# send the input to the device
		(image, target) = (image.to(_config.DEVICE), 
			target.to(_config.DEVICE))
		
		# make the predictions and calculate the validation
		# loss
		logit = model(image)
		loss = lossFunc(logit, target)
		
		totalTestLoss += loss.item()
		
		# output logits through the softmax layer to get output
		# predictions, and calculate the number of correct predictions
		pred = soft(logit)[0]*100
		testCorrect += (pred.argmax(dim=-1) == target).sum().item()

# print test data accuracy		
print(testCorrect/len(testDataset))

# initialize iterable variable
sweeper = iter(testLoader)

# grab a batch of test data
batch = next(sweeper)
(images, labels) = (batch[0], batch[1])

# initialize a figure
fig = plt.figure("Results", figsize=(10, 10 ))


# Get label with its translation
translate = translate.MEANING

translate = {x:translate[x] for x in testLoader.dataset.classes}

#print(translate)


# switch off autograd
with torch.no_grad():
	# send the images to the device
	images = images.to(_config.DEVICE)
	
	# make the predictions
	preds = model(images)
	
	# loop over all the batch
	for i in range(0, _config.PRED_BATCH_SIZE):
		# initialize a subplot
		ax = plt.subplot(_config.PRED_BATCH_SIZE, 1, i + 1)
		
		# grab the image, de-normalize it, scale the raw pixel
		# intensities to the range [0, 255], and change the channel
		# ordering from channels first tp channels last
		image = images[i]
		image = deNormalize(image).cpu().numpy()
		image = (image * 255).astype("uint8")
		image = image.transpose((1, 2, 0))
		
		# grab the ground truth label
		idx = labels[i].cpu().numpy()
		gtLabel = testDataset.classes[idx]

		
		# grab the predicted label
		pred = preds[i].argmax().cpu().numpy()
		predLabel = testDataset.classes[pred]

		# grab new translated class labels
		t_labels = list(translate.keys())[idx]
		
		
		# Create the info string, and add the results and image to the plot
		info = "Ground Truth: {} \n Predicted: {} \n Meaning: {}".format(
			gtLabel, 
			predLabel,
			translate[t_labels]
		)
		
		plt.imshow(image)
		plt.title(info, color='r')
		plt.axis("off")
	
	# show the plot
	plt.tight_layout()
	plt.show()

print(info)