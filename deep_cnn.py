import torch
import torch.nn as nn

# Define a deeper CNN model
class DeeperCNN(nn.Module):
    def __init__(self, num_classes=62, dropout_prob=0.5):
    # def __init__(self, num_classes=62):
        super(DeeperCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU()
        
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc2 = nn.Linear(128 * 32 * 32, 4096)
        self.relu7 = nn.ReLU()
        
        self.dropout1 = nn.Dropout(dropout_prob)  # Dropout layer
        
        self.fc1 = nn.Linear(4096, 4096)
        self.relu8 = nn.ReLU()
        
        self.dropout2 = nn.Dropout(dropout_prob)  # Dropout layer
        
        self.fc = nn.Linear(4096, num_classes)

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        
        x = self.pool1(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.relu4(x)
        
        x = self.pool2(x)
        
        x = self.conv5(x)
        x = self.relu5(x)
        
        x = self.conv6(x)
        x = self.relu6(x)
        
        x = self.pool3(x)
        
        # Flatten the feature maps
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc2(x)
        x = self.relu7(x)
        # x = self.dropout1(x)  # Apply dropout
        x = self.fc1(x)
        x = self.relu8(x)
        # x = self.dropout2(x)  # Apply dropout
        x = self.fc(x)
        
        return x



# Create an instance of the DeeperCNN model
model = DeeperCNN()

# Print the model architecture
print(model)

"""

from torchviz import make_dot


# Create an instance of your model
# model = DeeperCNN()

# Create a dummy input tensor (change dimensions to match your model)
input_tensor = torch.randn(1, 3, 128, 128)


# Forward pass to compute the graph
output_tensor = model(input_tensor)


# Generate the graph and save it as an image file
dot = make_dot(output_tensor, params=dict(model.named_parameters()))
dot.format = 'png'  # Specify the desired format
dot.render("model_graph.png")
dot.save('model_graph')
"""
"""

# Generate the graph
dot = make_dot(output_tensor, params=dict(model.named_parameters()))

# Display the graph in a Jupyter Notebook or save it as an image
# dot.view()
dot.save('model_graph_.png')
"""