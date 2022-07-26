from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1, box2,xyxy=True):
    """
    Returns the IoU of two bounding boxes


    """
    # Get the coordinates of bounding boxes
    if xyxy:  # bbox defined by corners
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[:,0] - box1[:,2] / 2, box1[:,0] + box1[:,2] / 2
        b1_y1, b1_y2 = box1[:,1] - box1[:,3] / 2, box1[:,1] + box1[:,3] / 2
        b2_x1, b2_x2 = box2[:,0] - box2[:,2] / 2, box2[:,0] + box2[:,2] / 2
        b2_y1, b2_y2 = box2[:,1] - box2[:,3] / 2, box2[:,1] + box2[:,3] / 2

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,
                                                                                     min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    # Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add the center offsets
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    # log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

    prediction[:, :, :4] *= stride

    return prediction


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = prediction.size(0)

    write = False

    for ind in range(batch_size):
        image_pred = prediction[ind]  # image Tensor
        # confidence threshholding
        # NMS

        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score) #crete a 7 dm vector with boundi box, obj confidence, most confident class and its score
        image_pred = torch.cat(seq, 1)

        non_zero_ind = (torch.nonzero(image_pred[:, 4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7) #remove predictions with low conf score
        except:
            continue

        if image_pred_.shape[0] == 0:
            continue
        #

        # Get the various (unique) classes detected in the image
        img_classes = unique(image_pred_[:, -1])  # -1 index holds the class index

        for cls in img_classes:
            # perform NMS

            # get the detections with one particular class
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            # sort the detections such that the entry with the maximum objectness
            # confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)  # Number of detections

            for i in range(idx):
                # Get the IOUs of all boxes that come after the one we are looking at
                # in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
                except ValueError:
                    break

                except IndexError:
                    break

                # Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i + 1:] *= iou_mask

                # Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(
                ind)  # Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    try:
        return output
    except:
        return 0


def letterbox_image(img, inp_dim):
    scaling,offset = resizeImg(img, inp_dim)

    # '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    # w, h = inp_dim
    scaling[0] = int(scaling[0]*img_w)
    scaling[1] = int(scaling[1] * img_h)
    scaling = scaling.astype(int)
    # new_w = int(img_w * min(w / img_w, h / img_h))
    # new_h = int(img_h * min(w / img_w, h / img_h))

    resized_image = cv2.resize(img, (scaling[0],scaling[1]), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    #canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image
    canvas[offset[1]:offset[1] + scaling[1], offset[0]:offset[0] + scaling[0], :] = resized_image

    return canvas

def resizeImg(img,inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    s = min(w / img_w, h / img_h)
    new_w = int(img_w * s)
    new_h = int(img_h * s)

    scaling = np.array([s,s])
    offset = np.array([(w - new_w) // 2,(h - new_h) // 2])

    return scaling, offset

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

#each target can have multiple objects for one pic (defined by img_idx). The loss should be computed on each image idnependently
#looping through each batch

#Converts targets to a vector of n_imagesx85x13x13
def buildTargets(targets,pred_maps):
    #targets = [img_idx,class_idx, bb_x,bb_y,bb_w,bb_h] in scaled image in R^nx6
    #each pred map in R^n_batchxgridxgridx(n_c+5)xn_anc

    num_objects = targets.shape[0]

    num_images = pred_maps[0].shape[0]
    num_classes = pred_maps[0].shape[3]-5

    true_img_idx = targets[:,0].int().cpu().numpy()
    true_class_idx = targets[:, 1].int().cpu().numpy()
    true_bb = targets[:,2:]

    #set indices for true classes for each object
    true_Classes = torch.zeros((num_objects,num_classes),device=targets.device)
    true_Classes[range(num_objects),true_class_idx] = 1

    obj_present = torch.ones((num_objects, 1), device=targets.device)

    Targets = []

    for map in pred_maps:
        grid_size = map.shape[2]

        Target = torch.zeros((num_images,grid_size,grid_size,num_classes + 5), device=targets.device)

        #true indices on grid with objects
        true_bb_ij = true_bb*grid_size
        true_bb_ij = true_bb_ij.int().cpu().numpy()
        true_bb_i,true_bb_j = true_bb_ij[:,0],true_bb_ij[:,1]

        true_vec = torch.hstack((true_bb*grid_size,obj_present,true_Classes))
        Target[true_img_idx,true_bb_i,true_bb_j,:] = true_vec.float()

        Targets.append(Target)

    return Targets

def GetLoss(Targets,model,inp_dim,CUDA):

    Anchors = model.anchors
    pred_maps = model.output_maps

    # Define different loss functions classification. It includes sigmoid for predictions
    BCEcls = nn.BCEWithLogitsLoss()
    BCEobj = nn.BCEWithLogitsLoss()
    MSE_loss = nn.MSELoss()

    Train_loss = 0.

    #map in batchxgridxgridx85x3
    for idx,map in enumerate(pred_maps):

        target = Targets[idx] #in batchxgidxgridx85
        anchors = Anchors[idx]

        grid_size = map.shape[1]
        num_anchors = len(anchors)
        stride = inp_dim // grid_size

        #-------------Losses for each anchor box

        #-- Loss on Objectiveness Score
        obj = target[...,4] == 1
        noobj = target[..., 4] == 0
        idx_obj = torch.argwhere(obj).squeeze(0)
        idx_noobj = torch.argwhere(noobj).squeeze(0)

        loss_noobj = 0.
        loss_classes = 0.
        loss_IoU = 0.
        loss_coord = 0.

        loss_anchor = 0.
        for na,a in enumerate(anchors):
            pred_obj_score = map[:,:,:,4,na]
            pred_classes = map[:,:,:,5:,na]
            pred_bbox = map[:, :, :, 0:4, na]
            target_bbox = target[:,:,:,0:4] #already multiplied by gird size

            num_objs = idx_obj.shape[0]

            anchor = [a[0]/stride,a[1]/stride]
            anchor = torch.FloatTensor(anchor)
            anchor = anchor.repeat(num_objs,1)

            if CUDA:
                anchor = anchor.cuda()

            #----------- 1) No Object Loss
            loss_noobj = loss_noobj+BCEobj(pred_obj_score[idx_noobj[:,0],idx_noobj[:,1],idx_noobj[:,2]],
                              target[idx_noobj[:,0],idx_noobj[:,1],idx_noobj[:,2],4])

            #-----Bounding box
            pred_bbox = pred_bbox[idx_obj[:, 0], idx_obj[:, 1], idx_obj[:, 2],:]
            target_bbox = target_bbox[idx_obj[:, 0], idx_obj[:, 1], idx_obj[:, 2], :]

            pred_bbox[:,0:2] = torch.sigmoid(pred_bbox[:,0:2])+idx_obj[:, 1:]
            pred_bbox[:, 2:] = torch.exp(pred_bbox[:, 2:]) * anchor

            #----------- 2) Object Loss IoU
            IoU = bbox_iou(target_bbox,pred_bbox,xyxy=False)
            loss_IoU = loss_IoU+(1.0 - IoU).mean()

            #----------- 3) Object Loss, Box Cooridnates
            loss_coord = loss_coord+MSE_loss(pred_bbox,target_bbox)

            # ----------- 4) Object Loss, Classes loss
            target_classes = target[idx_obj[:,0],idx_obj[:,1],idx_obj[:,2],5:]
            pred_classes = pred_classes[idx_obj[:, 0], idx_obj[:, 1], idx_obj[:, 2],:]
            loss_classes = loss_classes+BCEcls(pred_classes, target_classes)

            loss_anchor = loss_anchor+ 0.05*loss_IoU+0.05*loss_coord+1.0*loss_noobj+1*loss_classes

        loss = loss_anchor/num_anchors

        Train_loss = Train_loss+loss

    return Train_loss






