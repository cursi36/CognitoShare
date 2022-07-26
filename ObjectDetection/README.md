This folder contains a code and notes on **YOLO v3** for object detection.

A pretrained model on COCO dataset with 8' classes is provided in the folder *DarkNet/*

The folder contains python scripts for building a YOLO v3 model, for computing training loss, and for inference.

### Files:

- *detector.py* is the code to run predicitions, given the YOLO v3 model and some input images. The output image with predictions will be saved in a folder named *det/*
- *DarkNet.py* defines the model's architecture.
- *TestNetLoss.py* contains the code for computing the losses for YOLO v3 training.
 
