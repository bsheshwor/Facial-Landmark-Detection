# Facial Keypoint Detection

This mini project on **Facial Keypoints Detection** is on defining and training a convolutional neural network to perform facial keypoint detection, and using computer vision techniques to process images. 

Facial keypoints are shown by the small magenta dots shown on each of the faces in the image below. Facial Landmark Localization is the detection and localization of certain key regions or landmarks in a human’s face such as: eyes, nose, lips, and others. These keypoints are relevant for a variety of tasks, such as face filters, emotion recognition, pose recognition, and so on. 

Convolution Neural Network Architecture:


![I2](SS/I2.png)

![I3](SS/I3.png)

In each training and test image, there is a single face and 68 keypoints, with **coordinates (x, y)**, for that face. It can be shown in the image below:

![I4](SS/I4.png)

Let's look at some sample outputs:

![I1](SS/I1.png)

The code is able to look at any image, detect faces, and predict the locations of facial keypoints on the face.

After training the model, we were able to receive an average loss of 0.09344655024351004

The following graph shows the decreasing loss with each iteration of training:

![I4](SS/I5.png)
