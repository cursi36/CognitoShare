from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from utils import *
import argparse
import os
import os.path as osp
from Darknet import Darknet
import pickle as pkl
import pandas as pd
import random

import matplotlib.pyplot as plt

def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest='images', help=
    "Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)
    parser.add_argument("--det", dest='det', help=
    "Image / Directory to store detections to",
                        default="det", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help=
    "Config file",
                        default="DarkNet/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help=
    "weightsfile",
                        default="DarkNet/yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)

    return parser.parse_args()


args = arg_parse()
# images = "TrainingData/images/pic_1.png"
# im_original = cv2.imread("TrainingData/images/pic_1.png")
# truth = np.loadtxt("TrainingData/labels/pic_1.txt")

images = "TrainingData/images/train.jpg"
im_original = cv2.imread("TrainingData/images/train.jpg")
truth = np.loadtxt("TrainingData/labels/train.txt")

batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

num_classes = 80
classes = load_classes("data/coco.names")

# Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])

# If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()

# Set the model in evaluation mode
model.eval()

#
print("imgae size",im_original.shape)
im = prep_image(im_original,inp_dim)
im_scaling,im_offset = resizeImg(im_original,(inp_dim,inp_dim))

im = torch.cat((im,im),dim=0)
#
if CUDA:
    batch = im.cuda()
with torch.no_grad():
    prediction = model(batch, CUDA)
    pred_maps = model.output_maps
    print("prediction shape ", prediction.shape)

#first value is batch index
prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thesh)
prediction = prediction[0,1:]
bb_corners = prediction[0:4]
obj_score = prediction[4]
cls_prob = prediction[5]
cls_idx = prediction[6]

c1_pred = tuple(bb_corners[0:2].int().cpu().numpy())
c2_pred = tuple(bb_corners[2:4].int().cpu().numpy())

img = im[0,:].cpu().numpy()
img = img.transpose(1,2,0)*255
img = img.astype(np.uint8)
img = cv2.rectangle(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), c1_pred, c2_pred, (0,0,255), 4)

plt.figure()
plt.imshow(img)
plt.show()

#load true box in original picture shape

true_class = truth[0]
true_bb = truth[1:]
# s = np.array([im_original.shape[1],im_original.shape[0],im_original.shape[1],im_original.shape[0]])
s = np.array([im_original.shape[1]*im_scaling[0],im_original.shape[0]*im_scaling[1],
              im_original.shape[1]*im_scaling[0],im_original.shape[0]*im_scaling[1]])
true_bb = true_bb*s #bb center wrt original image
true_bb[0:2] = true_bb[0:2]+im_offset

bb_corner = np.copy(true_bb)
bb_corner[0] = (true_bb[0] - true_bb[2] / 2)
bb_corner[1] = (true_bb[1] - true_bb[3] / 2)
bb_corner[2] = (true_bb[0] + true_bb[2] / 2)
bb_corner[3] = (true_bb[ 1] + true_bb[3] / 2)

c1 = tuple(bb_corner[0:2].astype(int))
c2 = tuple(bb_corner[2:4].astype(int))
# img_true = cv2.rectangle(cv2.cvtColor(im_original, cv2.COLOR_BGR2RGB), c1, c2, (0,255,0), 4)
#
# img_true_scaled = letterbox_image(img_true, (inp_dim,inp_dim))
# img_true_scaled = cv2.rectangle(img_true_scaled, c1_pred, c2_pred, (0,0,255), 4)
#
# plt.figure()
# plt.imshow(img_true_scaled)
# plt.show()


img = cv2.rectangle(img, c1, c2, (0,255,0), 4)

plt.figure()
plt.imshow(img)
plt.show()

bbs_perc = true_bb/inp_dim

img_idx = np.array([0,0,1]).reshape(-1,1)
class_idx = np.array([6,6,6]).reshape(-1,1)
bbs = np.vstack((bbs_perc,bbs_perc,bbs_perc))
targets = np.hstack((img_idx,class_idx,bbs))
targets = torch.tensor(targets)
if CUDA:
    targets = targets.cuda()

Targets = buildTargets(targets,pred_maps)

GetLoss(Targets,model,inp_dim,CUDA)


torch.cuda.empty_cache()




