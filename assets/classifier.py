# import the necessary packages
from torch.nn import Linear
from torch.nn import Module


class Classifier(Module):
	def __init__(self, baseModel, numClasses, model):
		super().__init__()
		# initialize the base model 
		self.baseModel = baseModel
		
		# check if the base model is VGG, if so, initialize the FC
		# layer accordingly
		if model == "vgg":
			self.fc = Linear(baseModel.classifier[6].out_features, 
                                numClasses)
		
		# otherwise, the base model is of type ResNet so initialize
		# the FC layer accordingly
		else:
			self.fc = Linear(baseModel.fc.out_features, numClasses)

	def forward(self, x):
		# pass the inputs through the base model to get the features
		# and then pass the features through of fully connected layer
		# to get our output logits
		features = self.baseModel(x)
		logits = self.fc(features)
		
		# return the classifier outputs
		return logits
