
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


logger=logger_utils.get_logger('GTSRB_DNN_Logs')

######################## GTSRB Model #########################
class TrafficSignClassifier(nn.Module):
    def __init__(self, p=0.5):
        super(TrafficSignClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 43)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.maxpool1(x)
        x = self.dropout(x)
        
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.maxpool2(x)
        x = self.dropout(x)
        
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.maxpool3(x)
        x = self.dropout(x)
        
        x = x.view(-1, 128 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


 ############ Fetching Dataset ################
    
transform = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor()])

train_dataset = datasets.ImageFolder(root='D:/Code/Lab105_Personal/Atharva_MPD/Datasets/GTSRB/train_image', transform=transform)
logger.info(f"Length of training dataset: {len(train_dataset)}")
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4)

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
num_epochs = 5
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
    bootstrap_loader = DataLoader(train_dataset, sampler=bootstrap_sampler, batch_size=64)
    
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
save_dir = "/GTSRB_Ensemble_100/"
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
