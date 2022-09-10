import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.fc_drop1 = nn.Dropout(p=0.2)
        
        self.conv2 = nn.Conv2d(32, 36, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc_drop2 = nn.Dropout(p=0.2)
        
        self.conv3 = nn.Conv2d(36, 48, 5)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc_drop3 = nn.Dropout(p=0.2)
        
        self.conv4 = nn.Conv2d(48, 64, 3)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.fc_drop4 = nn.Dropout(p=0.2)
        
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.pool5 = nn.MaxPool2d(2, 2)
   
        self.fc6 = nn.Linear(64*4*4, 136)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.fc_drop1(x)
        
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.fc_drop2(x)
        
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.fc_drop3(x)
        
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.fc_drop4(x)
        
        x = self.pool5(F.relu(self.conv5(x)))
        x = x.view(x.size(0), -1)

        x = self.fc6(x)        
        return x
        