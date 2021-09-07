# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 12:08:59 2021

@author: DIV6OFO
"""
#from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import os
import numpy as np
import torch.nn.init
import random
import glob
import datetime
import tqdm

use_cuda = torch.cuda.is_available()

# parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
# parser.add_argument('--nChannel', metavar='N', default=100, type=int, 
#                     help='number of channels')
# parser.add_argument('--maxIter', metavar='T', default=1, type=int, 
#                     help='number of maximum iterations')
# parser.add_argument('--maxUpdate', metavar='T', default=1000, type=int, 
#                     help='number of maximum update count')
# parser.add_argument('--minLabels', metavar='minL', default=3, type=int, 
#                     help='minimum number of labels')
# parser.add_argument('--batch_size', metavar='bsz', default=1, type=int, 
#                     help='number of batch_size')
# parser.add_argument('--lr', metavar='LR', default=0.1, type=float, 
#                     help='learning rate')
# parser.add_argument('--nConv', metavar='M', default=2, type=int, 
#                     help='number of convolutional layers')
# parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int, 
#                     help='visualization flag')
# parser.add_argument('--input', metavar='FOLDERNAME',
#                     help='input image folder name', required=True)
# parser.add_argument('--stepsize_sim', metavar='SIM', default=1, type=float,
#                     help='step size for similarity loss', required=False)
# parser.add_argument('--stepsize_con', metavar='CON', default=5, type=float, 
#                     help='step size for continuity loss')
# args = parser.parse_args()

nChannel = 100
maxIter = 1
epochs = 10000
minLabels = 200
batch_size = 1
lr=0.001 #provare 0.001 + adam optimizer
nConv= 2
visualize = 2
inputFolder= "U:\Progetti\Sistemi di visione\ImgSegm\ImgForUnsSegm_train" #folder name
stepsize_sim = 1.0
stepsize_con = 5.0




# CNN model
class MyNet(nn.Module):
    def __init__(self,input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(nConv-1):
            self.conv2.append( nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(nChannel) )
        self.conv3 = nn.Conv2d(nChannel, nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

#img_list = sorted(glob.glob(inputFolder))
img_list=os.listdir(inputFolder)
print("immagini---", img_list)
im = cv2.imread(os.path.join(inputFolder, img_list[0]))
print(im.shape[2])

# train
model = MyNet( im.shape[2] )
if use_cuda:
    model.cuda()
model.train()

# similarity loss definition
loss_fn = torch.nn.CrossEntropyLoss()

# continuity loss definition
loss_hpy = torch.nn.L1Loss(size_average = True)
loss_hpz = torch.nn.L1Loss(size_average = True)

#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9) #provare Adam
optimizer = optim.Adam(model.parameters(), lr=lr)
label_colours = np.random.randint(255,size=(100,3))

for batch_idx in range(maxIter):
    print('Training started. '+str(datetime.datetime.now())+'   '+str(batch_idx+1)+' / '+str(maxIter))
    for im_file in range(int(len(img_list)/batch_size)):
        
        for loop in tqdm.tqdm(range(epochs)):
            
            im = []
            for batch_count in range(batch_size):
                # load image
                resized_im = cv2.imread(os.path.join(inputFolder,img_list[batch_size*im_file + batch_count]))
                resized_im = cv2.resize(resized_im, dsize=(224, 224)) #valutare l'input
                resized_im = resized_im.transpose( (2, 0, 1) ).astype('float32')/255.
                im.append(resized_im)

            data = torch.from_numpy( np.array(im) )
            if use_cuda:
                data = data.cuda()
            data = Variable(data)
    
            HPy_target = torch.zeros(data.shape[0], resized_im.shape[1]-1, resized_im.shape[2], nChannel)
            HPz_target = torch.zeros(data.shape[0], resized_im.shape[1], resized_im.shape[2]-1, nChannel)
            if use_cuda:
                HPy_target = HPy_target.cuda()
                HPz_target = HPz_target.cuda()

            # forwarding
            optimizer.zero_grad()
            output = model( data )
            output = output.permute( 0, 2, 3, 1 ).contiguous().view( data.shape[0], -1, nChannel )

            outputHP = output.reshape( (data.shape[0], resized_im.shape[1], resized_im.shape[2], nChannel) )
    
            HPy = outputHP[:, 1:, :, :] - outputHP[:, 0:-1, :, :]
            HPz = outputHP[:, :, 1:, :] - outputHP[:, :, 0:-1, :]    
            lhpy = loss_hpy(HPy,HPy_target)
            lhpz = loss_hpz(HPz,HPz_target)

            output = output.reshape( output.shape[0] * output.shape[1], -1 )
            ignore, target = torch.max( output, 1 )

            loss = stepsize_sim * loss_fn(output, target) + stepsize_con * (lhpy + lhpz)
            
            if loop%100 == 0:
                print("loss =", float    (loss))
            
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), os.path.join(inputFolder, 'b'+str(batch_size)+'_itr'+str(maxIter)+'_layer'+str(nConv+1)+'.pth'))
    

# model = model = MyNet(3)
# model.load_state_dict(torch.load("U:\Progetti\Sistemi di visione\ImgSegm\ImgForUnsSegm_train\b1_itr1_layer3.pth"))

label_colours = np.random.randint(255,size=(100,3))
#test_img_list = sorted(glob.glob(inputFolder+'/test/*'))
test_img_list = os.listdir("U:\Progetti\Sistemi di visione\ImgSegm\ImgForUnsSegm_test")
if not os.path.exists(os.path.join("U:\Progetti\Sistemi di visione\ImgSegm", 'result_color/')):
    os.mkdir(os.path.join("U:\Progetti\Sistemi di visione\ImgSegm", 'result_color/'))

print('Testing '+str(len(test_img_list))+' images.')
print(test_img_list)
for img_file in tqdm.tqdm(test_img_list):
    print(img_file)
   
    im = cv2.imread(os.path.join("U:\Progetti\Sistemi di visione\ImgSegm\ImgForUnsSegm_test", img_file))
    print(im)
    im = cv2.resize(im, dsize=(1500, 1500)) #valore massimo testato, aumentandoli ci sarà un problema di allocazione in memoria
    data = torch.from_numpy( np.array([im.transpose( (2, 0, 1) ).astype('float32')/255.]) )
    if use_cuda:
        data = data.cuda()
    data = Variable(data)
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, nChannel )
    ignore, target = torch.max( output, 1 )
    inds = target.data.cpu().numpy().reshape( (im.shape[0], im.shape[1]) )
    inds_rgb = np.array([label_colours[ c % nChannel ] for c in inds])
    inds_rgb = inds_rgb.reshape( im.shape ).astype( np.uint8 )
    cv2.imwrite( os.path.join("U:\Progetti\Sistemi di visione\ImgSegm", 'result_color/') + os.path.basename(img_file), inds_rgb )
    
print(len(inds_rgb[0][0]))

import cv2
from PIL import Image
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


#image= cv2.imread(os.path.join("U:\Progetti\Sistemi di visione\ImgSegm", 'result_color/', img_list[0]))
image= cv2.imread(r"U:\Progetti\Sistemi di visione\ImgSegm\result_color\Adam_0.001_1000labels.png")
print(image)

# print(image)
# output=image.copy()
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# minDist = 100
# param1 = 30 #500
# param2 = 50 #200 #smaller value-> more false circles
# minRadius = 5
# maxRadius = 20 #10

# circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

# if circles is not None:
#     circles = np.uint16(np.around(circles))
#     for i in circles[0,:]:
#         cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)

# # Show result for testing:

# cv2.imwrite(os.path.join("U:\Progetti\Sistemi di visione\ImgSegm", 'result/')+ ".jpg", image)



hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
green_lower = np.array([(297,87,79)], np.uint8)
green_upper = np.array([102, 200, 255], np.uint8)
mask = cv2.inRange(hsv, green_lower, green_upper)



#mask = cv2.inRange(hsv, (296,86,78), (299,89,81))
result = cv2.bitwise_and(image, image, mask=mask)
plt.imshow(result)

plt.show()

print(mask)


import cv2
import numpy as np

inputFolder= "U:\Progetti\Sistemi di visione\Dataset_hackaton\Hackaton_A"

image= cv2.imread(r"U:\Progetti\Sistemi di visione\Dataset_hackaton\Hackaton_A\A_rotor (6).png")
print(image)

output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_blurred = cv2.blur(gray, (3,3))
circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=15, maxRadius=30)

#print(circles)
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(output, (x-5, y-5), (x+5, y+5), (0,128,255), -1)
        
        cv2.imshow("output", np.hstack([image, output]))
        cv2.waitKey(0)

print(circles)
# image_color= cv2.imread(r"U:\Progetti\Sistemi di visione\Dataset_hackaton\Hackaton_B\B_rotor (5).png")
# image_ori = cv2.cvtColor(image_color,cv2.COLOR_BGR2GRAY)      
        
# lower_bound = np.array([0,0,10])
# upper_bound = np.array([255,255,195])

# image = image_color

# mask = cv2.inRange(image_color, lower_bound, upper_bound)

# # mask = cv2.adaptiveThreshold(image_ori,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
# #             cv2.THRESH_BINARY_INV,33,2)

# kernel = np.ones((3, 3), np.uint8)

# #Use erosion and dilation combination to eliminate false positives. 
# #In this case the text Q0X could be identified as circles but it is not.
# mask = cv2.erode(mask, kernel, iterations=6)
# mask = cv2.dilate(mask, kernel, iterations=3)

# closing = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
#         cv2.CHAIN_APPROX_SIMPLE)[0]
# contours.sort(key=lambda x:cv2.boundingRect(x)[3])

# array = []
# ii = 1

# for c in contours:
#     (x,y),r = cv2.minEnclosingCircle(c)
#     center = (int(x),int(y))
#     r = int(r)
#     if r >= 6 and r<=10:
#         cv2.circle(image,center,r,(0,255,0),2)
#         array.append(center)

# cv2.imshow("preprocessed", image_color)
# cv2.waitKey(0)



from skimage.io import imread, imshow 
from skimage.feature import match_template
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import numpy as np

image= cv2.imread(r"U:\Progetti\Sistemi di visione\ImgSegm\MaskedImages\test.jpg")
print(image)

output = image.copy()
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray=mask
template = cv2.imread(r"U:\Progetti\Sistemi di visione\ImgSegm\TemplateMatchingAlg\pattern.png")

from skimage.feature import match_template
result = match_template(mask, template)
#imshow(result, cmap='viridis')

x, y = np.unravel_index(np.argmax(result), result.shape)
print((x, y))

template_width, template_height = template.shape
rect = plt.Rectangle((y, x), template_height, template_width, 
                         color='b', fc='none')
plt.gca().add_patch(rect)


print(template.size)
print(mask.size)
          
	