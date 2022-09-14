import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2

import numpy as np

# Video Capture
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print('Unable to open camera.')

frame_width = int(camera.get(3))
frame_height = int(camera.get(4))

# print(frame_width,frame_height)

# vidout = cv2.VideoWriter('COMP484/Facial-Landmark-Detection/CV/outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

size = 224
track_xy=[208,128]
mean = [track_xy[0]+size/2,track_xy[1]+size/2]
mean_landmark = [0,0]
face_map = [[0,0],[0,0]]    # start point(x,y) # end point (x,y)

cutout = [*range(0,track_xy[0])]
cutout.extend([*range(track_xy[0]+size,640)])

def crop(image):
    out = image.copy()
    out = np.delete(out, cutout, axis = 1)
    try:
        a = out.shape[2]
        out = out[int(track_xy[1]):int(track_xy[1])+size]
    except:
        out = out[int(track_xy[1]):int(track_xy[1])+size]/255.0
    return out



# Neural Network

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
FLDmodel.load_state_dict(torch.load('keypoints_model_1.pt'))
FLDmodel.eval()

def show(image, key_pts=None):
    """Show image with keypoints"""
    plt.imshow(image)
    if key_pts is not None:
        plt.scatter(key_pts[0], key_pts[1], s=20, marker='.', c='m')
    plt.show()

def out(image):
    global mean
    global cutout
    global face_map
    #resizing
    # image = cv2.resize(image,224,224)
    
    #randomcrop of 224x224

    # Normalization
    # image = np.copy(image)

    # #converting images into grayscale
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # #normalizing the images which is of 255px each
    # image = image/255.0

    # To Tensor
    
    if(len(image.shape)==2):
        image = image.reshape(image.shape[0], image.shape[1],1)

    image = image.transpose((2,0,1))
    torchimg = torch.from_numpy(image)

    # print(torchimg.size())

    torchimg = torchimg.type(torch.FloatTensor)
    output = FLDmodel(torchimg)
    output = output.view(output.size()[0], 68, -1)
    output = output.detach().numpy()*50+100
    output = np.transpose(output[0]).tolist()
    face_map = [[track_xy[0]+np.min(output[0]),track_xy[1]+np.min(output[1])],[track_xy[0]+np.max(output[0]),track_xy[1]+np.max(output[1])]]
    m = [track_xy[0]+np.sum(output[0])/68.0,track_xy[1]+np.sum(output[1])/68.0]
    mean_landmark[0] = int(m[0])
    mean_landmark[1] = int(m[1])
    track_xy[0] = track_xy[0] + mean_landmark[0] - mean[0]
    if track_xy[0] < 0: 
        track_xy[0] = 0
    elif track_xy[0] > 416:
        track_xy[0] = 416
    track_xy[1] = track_xy[1] + mean_landmark[1] - mean[1] 
    if track_xy[1] < 0: 
        track_xy[1] = 0
    elif track_xy[1] > 256:
        track_xy[1] = 256
    mean = mean_landmark.copy()  
    cutout = [*range(0,int(track_xy[0]))]
    cutout.extend([*range(int(track_xy[0])+size,640)])
    return output


if __name__=="__main__":
    # og = mpimg.imread('COMP484/Facial-Landmark-Detection/CV/sajan.jpg')
    while(True):
        ret, frame = camera.read()
        
        if ret == True:
        # Write the frame into the file 'output.avi'
            # out.write(frame)
            # Display the resulting frame  
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) 
            normal = crop(gray)
            cropped = frame.copy()
            cropped = crop(cropped)
            face = frame.copy()
            output = out(normal)
            for i,j in zip(output[0],output[1]):
                frame = cv2.circle(frame,(int(track_xy[0])+int(i),int(track_xy[1])+int(j)),1,(0,225,0),-1)
                cropped = cv2.circle(cv2.UMat(cropped),(int(i),int(j)),1,(250,0,0),-1)
                face = cv2.rectangle(face,(int(face_map[0][0]),(int(face_map[0][1]))),(int(face_map[1][0]),int(face_map[1][1])),(100,255,100),2)

            cv2.imshow('frame',frame)
            cv2.imshow('cropped',cropped)
            cv2.imshow('face_map',face)
            # Press Q on keyboard to stop recording
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            # Break the loop
        else:
            break 

camera.release()
# vidout.release()
cv2.destroyAllWindows()

