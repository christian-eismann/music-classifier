import torch
from torch import nn

# model definition
class CNN(nn.Module):
     def __init__(self):
         super(CNN, self).__init__()

         self.cnn_layers = nn.Sequential(
             nn.Conv2d(2 ,   16, kernel_size=4, padding=2), nn.Tanh(), 
             nn.MaxPool2d(2), nn.BatchNorm2d(16), # 128
             nn.Conv2d(16,   32, kernel_size=4, padding=2), nn.Tanh(), 
             nn.MaxPool2d(2), nn.BatchNorm2d(32), # 64
             nn.Conv2d(32,   64, kernel_size=4, padding=2), nn.Tanh(), 
             nn.MaxPool2d(2), nn.BatchNorm2d(64),  # 32
             nn.Conv2d(64,  128, kernel_size=4, padding=2), nn.Tanh(), 
             nn.MaxPool2d(2), nn.BatchNorm2d(128), # 16
             nn.Conv2d(128, 256, kernel_size=4, padding=2), nn.Tanh(), 
             nn.MaxPool2d(2), nn.BatchNorm2d(256),  # 8
             nn.Flatten()
         )

         self.linear_layers = nn.Sequential(
             nn.Linear(8*8*256, 2048), nn.Tanh(),
             nn.Linear(2048, 2048), nn.Tanh(),
             nn.Linear(2048, 10)
         )

     def forward(self, x):
         x = self.cnn_layers(x)
         x = x.view(x.size(0), -1)
         x = self.linear_layers(x)
         return x
