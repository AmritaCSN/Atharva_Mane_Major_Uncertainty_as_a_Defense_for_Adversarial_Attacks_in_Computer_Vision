
##################### Imports ############################

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import TensorDataset, DataLoader,SubsetRandomSampler
from tqdm import tqdm  
from torchvision import transforms, datasets
import os
import cv2
import logger_utils


logger=logger_utils.get_logger('MNIST_DNN_Logs')

######################## MNIST Model #########################
class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  # input channels=1 (grayscale), output channels=32, kernel_size=3x3
        self.conv2 = nn.Conv2d(32, 32, 3) # input channels=32, output channels=32, kernel_size=3x3
        self.maxpool1 = nn.MaxPool2d(2)   # kernel_size=2x2, default stride=2
        self.conv3 = nn.Conv2d(32, 64, 3) # input channels=32, output channels=64, kernel_size=3x3
        self.conv4 = nn.Conv2d(64, 64, 3) # input channels=64, output channels=64, kernel_size=3x3
        self.maxpool2 = nn.MaxPool2d(2)   # kernel_size=2x2, default stride=2
        self.fc1 = nn.Linear(64 * 4 * 4, 512) # 64 channels * 4x4 feature map size, output features=512
        self.fc2 = nn.Linear(512, 10)     # 512 input features, output classes=10

    def forward(self, x):
        x = F.relu(self.conv1(x))         # Conv1 -> ReLU
        x = F.relu(self.conv2(x))         # Conv2 -> ReLU
        x = self.maxpool1(x)              # MaxPool1
        x = F.relu(self.conv3(x))         # Conv3 -> ReLU
        x = F.relu(self.conv4(x))         # Conv4 -> ReLU
        x = self.maxpool2(x)              # MaxPool2
        x = x.view(-1, 64 * 4 * 4)        # Flatten before FC layer
        x = F.relu(self.fc1(x))           # FC1 -> ReLU
        x = self.fc2(x)                   # FC2 (no activation applied here, will be handled in the loss function)
        return x


 ############ Fetching Dataset ################
    
transform = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor()])

train_dataset = datasets.ImageFolder(root='D:/Code/Lab105_Personal/Atharva_MPD/Datasets/GTSRB/train_image', transform=transform)
logger.info(f"Length of training dataset: {len(train_dataset)}")
train_loader = DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True, num_workers=4)

 ############ Fetching GPU ################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(device)

start=time.strftime("%Y-%m-%d %H:%M:%S")
logger.info(f"Code started at: {start}")

######################## Training #########################

logger.info("####################################################")
logger.info("############ Starting Training Phase ############")
logger.info("####################################################")
# Define the number of models and bootstrap samples
n_models = 100
num_epochs = 15 
# List to store the trained models
ensemble_models = []

# Loop to create and train ensemble models
for model_index in range(n_models):
    # Create a new instance of the base model
    model = TrafficSignClassifier()
    
    # Check if GPU is available and move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Create a bootstrap sample
    bootstrap_indices = np.random.choice(len(train_dataset), size=len(train_dataset), replace=True)
    bootstrap_sampler = SubsetRandomSampler(bootstrap_indices)
    bootstrap_loader = DataLoader(train_dataset, sampler=bootstrap_sampler, batch_size=2000)
    
    # Create a tqdm progress bar for the epochs
    for epoch in tqdm(range(num_epochs), desc=f'Training Model {model_index + 1}/{n_models}'):
        running_loss = 0.0  # Initialize running loss for the epoch

        # Iterate over the training data loader
        for inputs, labels in bootstrap_loader:
            # Move inputs and labels to the GPU if available
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            # Add the batch loss to the running loss for the epoch
            running_loss += loss.item()

        # Print the average loss for the epoch
        epoch_loss = running_loss / len(bootstrap_loader)
        logger.info(f"Model {model_index + 1}/{n_models}, Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    # Append the trained model to the ensemble
    ensemble_models.append(model)
    
logger.info("------------Saving the Models------------")
# Define the directory where you want to save the models
save_dir = "MNIST_Ensemble/"
# Ensure the directory exists, if not create it
import os
os.makedirs(save_dir, exist_ok=True)
# Loop through each model in the ensemble and save it
for i, model in enumerate(ensemble_models):
    model_path = os.path.join(save_dir, f"model_{i+1}.pth")
    torch.save(model.state_dict(), model_path)
end=time.strftime("%Y-%m-%d %H:%M:%S")
logger.info(f"Code completed at: {end}")

logger.info("####################################################")
logger.info("############ Code Run Complete ############")
logger.info("####################################################")