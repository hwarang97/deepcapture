import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from torchinfo import summary

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Define the layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(128 * 19 * 19, 128)
        self.dropout2 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x