
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchcam.methods import GradCAM, GradCAMpp
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

from deep_cnn import DeeperCNN

# Load the pretrained CNN model (e.g., VGG16)
# model = torch.hub.load('pytorch/vision', 'vgg16', pretrained=True)
# VGG
# model=models.vgg16(pretrained=True)
# model.eval()
# # Select the layer just before the final classification layer
# target_layer = model.features[-1]


# # ResNet
# model = models.resnet50(pretrained=True)
# model.eval()
# target_layer = model.layer4[-1]

# CNN
model = DeeperCNN()
model.eval()
target_layer = model.relu6


# Load and preprocess an image
img_path = 'dataset/val/Akoben/Akoben_original_Akoben_08.png_0deaf18c-9c75-4939-b121-5a21ed0c3853.png'

preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open(img_path)
img = preprocess(img).unsqueeze(0)  # Add a batch dimension

# Create a hook to retrieve activations from the target layer
activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

target_layer.register_forward_hook(get_activation('target_layer'))

# Forward pass to get the feature maps
with torch.no_grad():
    output = model(img)

# Get the predicted class index
predicted_class = torch.argmax(output, dim=1).item()

# Get the activations from the target layer
feature_maps = activation['target_layer'].squeeze()

# Calculate the heatmap
heatmap = torch.mean(feature_maps, dim=0)
heatmap = F.relu(heatmap)
heatmap /= torch.max(heatmap)

# Resize the heatmap to match the dimensions of the original image
heatmap = heatmap.cpu().numpy()
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.resize(heatmap, (img.size(3), img.size(2)))

# Normalize heatmap values between 0 and 255
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255

# Convert heatmap to a grayscale format (CV_8U)
heatmap = np.uint8(heatmap)

# Apply the color map
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Convert the original image to the same data type as the heatmap
original_img = cv2.cvtColor(np.array(img.squeeze().permute(1, 2, 0)), cv2.COLOR_RGB2BGR)
original_img = original_img.astype(np.uint8)

# Overlay heatmap on the original image
overlayed_img = cv2.addWeighted(original_img, 0.7, heatmap_color, 0.3, 0)

# Display the original image, heatmap, and overlayed image
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img.squeeze().permute(1, 2, 0))
axes[0].set_title('Original Image')
axes[1].imshow(heatmap)
axes[1].set_title('Heatmap')
axes[2].imshow(overlayed_img)
axes[2].set_title('Overlayed Image')
plt.show()


# import matplotlib.pyplot as plt

# # Create a figure and axis for plotting
# fig, ax = plt.subplots()

# # Define the text with LaTeX color formatting
# formatted_text = r"My name is \textcolor{red}{Alice} and I am 30 years old."

# # Plot the text on the axis
# ax.text(0.5, 0.5, formatted_text, fontsize=12, ha="center", va="center", usetex=True)

# # Remove axis labels and ticks
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_xlabel("")
# ax.set_ylabel("")

# # Show the plot
# plt.show()




