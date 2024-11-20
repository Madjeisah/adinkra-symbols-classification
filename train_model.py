# USAGE
# python train_model.py --model vgg
# python train_model.py --model resnet
# Some codes were borrowed from:
# https://medium.com/swlh/deep-learning-for-image-classification-creating-cnn-from-scratch-using-pytorch-d9eeb7039c12

# import the necessary packages
from assets import _config
from assets.classifier import Classifier
from assets.datapipeline import get_dataloader 
from assets.datapipeline import train_val_split
from deep_cnn import DeeperCNN

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, auc
# from sklearn.metrics import roc_curve, roc_auc_score, auc

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import Normalize
from torch.nn import CrossEntropyLoss
from torch.nn import Softmax
from torch import optim
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import torch
import time

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="vgg",
	choices=["cnn","vgg", "resnet"], help="name of the backbone model")

args = vars(ap.parse_args())

# check if the name of the backbone model is CNN
if args["model"] == "cnn":
	# load our CNN model
	baseModel = DeeperCNN(num_classes=62, dropout_prob=0.5)

# Or, the backbone model we will be using is a VGG
elif args["model"] == "vgg":
	# load VGG-11 model
	baseModel = torch.hub.load("pytorch/vision:v0.10.0", "vgg11",
		weights=True, skip_validation=True)
	
	# freeze the layers of the VGG-11 model
	for param in baseModel.features.parameters():
		param.requires_grad = False

# otherwise, the backbone model we will be using is a ResNet
elif args["model"] == "resnet":
	# load ResNet 18 model
	baseModel = torch.hub.load("pytorch/vision:v0.10.0", "resnet18",
		weights=True, skip_validation=True)
	
	# define the last and the current layer of the model
	lastLayer = 8
	currentLayer = 1

	# loop over the child layers of the model
	for child in baseModel.children():
		
		# check if we haven't reached the last layer
		if currentLayer < lastLayer:
			
			# loop over the child layer's parameters and freeze them
			for param in child.parameters():
				param.requires_grad = False
		
		# otherwise, we have reached the last layers so break the loop
		else:
			break
		
		# increment the current layer
		currentLayer += 1  

# define the transform pipelines
imgTransform = Compose([
	RandomResizedCrop(_config.IMAGE_SIZE),
	ToTensor(),
	Normalize(mean=_config.MEAN, std=_config.STD)
])

# create training and test dataset using ImageFolder
trainDataset = ImageFolder(_config.TRAIN_PATH, imgTransform)
testDataset = ImageFolder(_config.TEST_PATH, imgTransform)

# Split trainDataset to create training and validation data
(trainDataset, valDataset) = train_val_split(dataset=trainDataset)

# create training and validation data loaders 
trainLoader = get_dataloader(trainDataset, _config.BATCH_SIZE)
valLoader = get_dataloader(valDataset, _config.BATCH_SIZE)

# create test data loaders
testLoader = get_dataloader(testDataset, _config.PRED_BATCH_SIZE)

print('Train: ', len(trainDataset))
print('Val: ', len(valDataset))
print('Test: ', len(testDataset))

# build the custom model
model = Classifier(baseModel=baseModel.to(_config.DEVICE),
	numClasses=62, model=args["model"])
model = model.to(_config.DEVICE)

# initialize loss function and optimizer
lossFunc = CrossEntropyLoss()
lossFunc.to(_config.DEVICE)
optimizer = optim.Adam(model.parameters(), lr=_config.LR)

# initialize the softmax activation layer
softmax = Softmax()

# calculate steps per epoch for training and validation set
trainSteps = len(trainDataset) // _config.BATCH_SIZE
valSteps = len(valDataset) // _config.BATCH_SIZE

# initialize a dictionary to store training history
H = {
	"trainLoss": [],
	"trainAcc": [],
	"valLoss": [],
	"valAcc": []
}


# Initiate start time for training
start_time = time.time()

# Loop over epochs to train the network
print()
print("[INFO] training the network...")
for epoch in range(_config.EPOCHS):
	# set the model in training mode
	model.train()
	
	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalValLoss = 0
	
	# initialize the number of correct predictions in the training
	# and validation step
	trainCorrect = 0
	valCorrect = 0

	# loop over the training set
	for (image, target) in tqdm(trainLoader):
		
		# send the input to the device
		(image, target) = (image.to(_config.DEVICE),
			target.to(_config.DEVICE))
		
		# perform a forward pass and calculate the training loss
		logits = model(image)
		loss = lossFunc(logits, target)
		
		# zero out the gradients, perform the backpropagation step,
		# and update the weights
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		# add the loss to the total training loss so far, pass the
		# output logits through the softmax layer to get output
		# predictions, and calculate the number of correct predictions
		totalTrainLoss += loss.item()
		pred = softmax(logits)
		trainCorrect += (pred.argmax(dim=-1) == target).sum().item()

	# switch off autograd
	with torch.no_grad():
		# set the model in evaluation mode
		model.eval()
		
		# loop over the validation set
		for (image, target) in tqdm(valLoader):
			
			# send the input to the device
			(image, target) = (image.to(_config.DEVICE),
				target.to(_config.DEVICE))
			
			# make the predictions and calculate the validation
			# loss
			logits = model(image)
			valLoss = lossFunc(logits, target)
			totalValLoss += valLoss.item()
			
			# pass the output logits through the softmax layer to get
			# output predictions, and calculate the number of correct
			# predictions
			pred = softmax(logits)
			valCorrect += (pred.argmax(dim=-1) == target).sum().item()

	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgValLoss = totalValLoss / valSteps
	
	# calculate the training and validation accuracy
	trainCorrect = trainCorrect / len(trainDataset)
	valCorrect = valCorrect / len(valDataset)
	
	# update our training history
	H["trainLoss"].append(avgTrainLoss)
	H["valLoss"].append(avgValLoss)
	H["trainAcc"].append(trainCorrect)
	H["valAcc"].append(valCorrect)
	
	# print the model training and validation information
	print(f"[INFO] EPOCH: {epoch + 1}/{_config.EPOCHS}")
	print(f"Train loss: {avgTrainLoss:.6f}, Train accuracy: {trainCorrect:.4f}")
	print(f"Val loss: {avgValLoss:.6f}, Val accuracy: {valCorrect:.4f}")


# Testing loop (evaluate on the test set)
model.eval()
with torch.no_grad():
    total_correct = 0
    total_samples = 0
    all_predicted = []
    all_target = []
    all_probabilities = []
    
    for (image, target) in tqdm(testLoader):
        # send the input to the device
    	(image, target) = (image.to(_config.DEVICE),target.to(_config.DEVICE))
     	logits = model(image)
    	_, predicted = torch.max(logits, 1)
    	total_samples += target.size(0)
    	total_correct += (predicted == target).sum().item()

    	# append predicted and target tensors
    	all_predicted.append(predicted.cpu())
    	all_target.append(target.cpu())

    	# obtain class probabilities
    	probabilities = torch.softmax(logits, dim=1).cpu().numpy()

    	# append target and class probabilities
    	# all_target.extend(target.cpu().numpy())
    	all_probabilities.extend(probabilities)

    # Concatenate predicted and target tensors
    all_predicted = torch.cat(all_predicted)
    all_target = torch.cat(all_target)

test_accuracy = total_correct / total_samples

print()
print(f'Overall Test Accuracy: {test_accuracy * 100:.2f}%')
end_time = time.time()
print("Execution time: {:.3f}".format(end_time-start_time, "secs"))


# Calculate and print the classification report
classification_rep = classification_report(all_target, all_predicted)
print("Classification Report:")
print(classification_rep)


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["trainLoss"], label="train_loss")
plt.plot(H["valLoss"], label="val_loss")
plt.plot(H["trainAcc"], label="train_acc")
plt.plot(H["valAcc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(_config.PLOT_PATH+args["model"]+'-'+str(_config.EPOCHS)+'.png')


"""
# ///////////////////////////////////////
# Compute ROC curve and ROC-AUC scores
# ///////////////////////////////////////
# Initialize dictionaries to store FPR, TPR, and ROC AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

all_probabilities = np.array(all_probabilities)
n_classes = len(testDataset.classes[10:20])

# Initialize a tensor to store binary labels
binary_labels = []

for class_idx in range(n_classes):
    
    # Create a binary tensor where elements equal to the current class index are set to 1
    class_labels = (all_target == class_idx).cpu().numpy()

    # Append the binary tensor to the list
    binary_labels.append(class_labels)
    
    # Compute ROC curve and ROC AUC for the binary classification
    fpr[class_idx], tpr[class_idx], _ = roc_curve(binary_labels[class_idx], 
    	all_probabilities[:, class_idx])
    roc_auc[class_idx] = auc(fpr[class_idx], tpr[class_idx])

# Plot ROC curves for each class
plt.figure(figsize=(8, 6))
# colors = ['darkorange', 'navy', 'green', 'red', 'purple', 'brown']  # Add more colors as needed

for class_idx in range(n_classes):
    plt.plot(fpr[class_idx], tpr[class_idx], lw=2,
             label=f'Class {class_idx} (AUC = {roc_auc[class_idx]:.2f})')

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC-AUC Curve')
plt.legend(loc="lower right")
plt.show()

"""
# serialize the model state to disk
torch.save(model.state_dict(), _config.MODEL_PATH+args["model"]+'-'+str(_config.EPOCHS)+'.pth')
print()

