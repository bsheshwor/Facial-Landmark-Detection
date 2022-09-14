import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2

import numpy as np

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
        # x = x.view(x.size(0), -1)

        x = x.view(1,-1)

        x = self.fc6(x)        
        return x

FLDmodel = Net()
FLDmodel.load_state_dict(torch.load('COMP484/Facial-Landmark-Detection/keypoints_model_1.pt'))
FLDmodel.eval()

def show(image, key_pts=None):
    """Show image with keypoints"""
    plt.imshow(image)
    if key_pts is not None:
        plt.scatter(key_pts[0], key_pts[1], s=20, marker='.', c='m')
    plt.show()

def tupletoarray(output):
    data = [[],[]]
    for i in output[0]:
        data[0].append(i[0])
        data[1].append(i[1])
    return data

def out(image):
    #resizing
    # image = cv2.resize(image,224,224)
    
    #randomcrop of 224x224

    # Normalization
    image = np.copy(image)

    #converting images into grayscale
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #normalizing the images which is of 255px each
    image = image/255.0

    # To Tensor
    
    if(len(image.shape)==2):
        image = image.reshape(og.shape[0], og.shape[1],1)

    image = image.transpose((2,0,1))
    torchimg = torch.from_numpy(image)

    # print(torchimg.size())

    torchimg = torchimg.type(torch.FloatTensor)
    output = FLDmodel(torchimg)
    output = output.view(output.size()[0], 68, -1)
    output = output.detach().numpy()*50+100
    output = tupletoarray(output)
    return output


if __name__=="__main__":
    og = mpimg.imread('COMP484/Facial-Landmark-Detection/CV/sajan.jpg')

    output = out(og.copy())
    show(og,output)