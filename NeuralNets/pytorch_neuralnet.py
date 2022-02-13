# -*- coding: utf-8 -*-
"""NeuralNet_Classifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PBKLJIP3XMjd2Q7wrxLPhOMRhHvn5u8B
"""

# Commented out IPython magic to ensure Python compatibility.
# %cd "/content/drive/MyDrive/Colab Notebooks/Classification_ML/"

"""**Classify MNIST dataset**"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets 
import cv2 as cv
from google.colab.patches import cv2_imshow
import numpy as np
import matplotlib.pyplot as plt
import random 

if (torch.cuda.device_count()):
  print(torch.cuda.device_count())
  print(torch.cuda.get_device_name(0))

#Assign cuda GPU located at location '0' to a variable
  cuda0 = torch.device('cuda:0')
else:
  cuda0 = torch.device('cpu')

train_data = dsets.MNIST(root="",train=True,download=True, transform=transforms.ToTensor())
val_data = dsets.MNIST(root="",train=False,download=True, transform=transforms.ToTensor())

data = train_data[0]
img = np.transpose(data[0].numpy(), (1, 2, 0))
img = img[:,:,0] #for gray scale only "D array needed"
plt.imshow(img,cmap='gray')
print("Value",data[1] )

#Create the classifier. In this case just one hidden layer with linear activation
class NetClassifier(nn.Module):
  def __init__(self,in_size,hidden_sizes,out_size):
    super(NetClassifier,self).__init__()
    self.in_size = in_size

    self.act = torch.nn.ReLU()
    
    self.input = nn.Linear(in_size,hidden_sizes[0])

    self.hidden = torch.nn.ModuleList()

    for i in range(len(hidden_sizes)-1):
      hidden = nn.Linear(hidden_sizes[i],hidden_sizes[i+1])
      self.hidden.add(hidden)

    self.output = nn.Linear(hidden_sizes[-1],out_size)

    self.softmax = nn.Softmax()

  def forward(self,x):

    x = self.input(x)
    x = self.act(x)

    for h in self.hidden:
      x = h(x)
      x = self.act(x)

    out = self.output(x)
    proba = self.softmax(out)
    _,predicted_class = torch.max(out, 1)

    return out,proba,predicted_class

class ConvNetClassifier(nn.Module):
  def __init__(self,image_size,kernel_size,out_size,in_channels=1,out_channels=1):
    super(ConvNetClassifier,self).__init__()

    self.ConvNets = nn.ModuleList()
    self.FCNets = nn.ModuleList()

    self.ConvNets.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size))
    image_size = self.getConvOutDim(image_size,kernel_size)

    self.ConvNets.append(nn.MaxPool2d(2))
    image_size = self.getConvOutDim(image_size,2,2)

    self.ConvNets.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size))
    image_size = self.getConvOutDim(image_size,kernel_size)

    self.ConvNets.append(nn.MaxPool2d(2))
    image_size = self.getConvOutDim(image_size,2,2)

    in_size = out_channels*image_size[0]*image_size[1]
    self.FCNets.append(nn.Linear(in_size,100))
    self.FCNets.append(nn.Linear(100,50))

    self.outLayer = nn.Linear(50,out_size)

    self.act = nn.ReLU()

    self.softmax = nn.Softmax()

  def getConvOutDim(self,image_size,kernel_size,stride=1):
        h_out = int((image_size[0]-kernel_size)/stride)+1
        w_out = int((image_size[1]-kernel_size)/stride)+1

        return (h_out,w_out)

  
  #input is NxCxHxW
  def forward(self,x):

    for idx,net in enumerate(self.ConvNets):
      x = net(x)

    x = torch.flatten(x, 1)

    for net in self.FCNets:
      x = net(x)
      x = self.act(x)

    out = self.outLayer(x)

    proba = self.softmax(out)
    _,predicted_class = torch.max(out, 1)

    return out,proba,predicted_class

def TrainNet(Net,Data_train,Data_val,ConvNet=False,lr=1e-03,Epochs = 1000):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Net.parameters(), lr=lr)

    N_batches = len(Data_train)

    for epoch in range(Epochs):
      loss_tot = 0
      for x,y in Data_train:
        x = x.to(cuda0)
        y = y.to(cuda0)
        if not ConvNet:
          y_pred = Net(x.view(-1,Net.in_size))[0]
        else:
          y_pred = Net(x)[0]

        loss = loss_function(y_pred,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_tot = loss_tot + loss.item()
      
      loss_tot = loss_tot/N_batches
      print("*****epoch: ",epoch," loss: ",loss_tot)

      #Validation accuracy:
      correct = 0
      for x,y in Data_val:
        x = x.to(cuda0)
        y = y.to(cuda0)
        
        if not ConvNet:
          Net_eval = Net.eval()
          _,_,pred_class = Net_eval(x.view(-1,Net.in_size))
        else:
          _,_,pred_class = Net(x)
        correct = correct+(pred_class==y).sum().item()

      accuracy = correct/(y.shape[0])
      print("*****accuracy: ",accuracy)

in_size = img.shape[0]*img.shape[1] #squeeze the image
print(in_size)
out_size = 10 #10 classes form 0 to 9

train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=100)
val_loader = torch.utils.data.DataLoader(dataset=val_data,batch_size=len(val_data))

h_sizes = [10]
model = NetClassifier(in_size,h_sizes,out_size)
model.to(cuda0)

TrainNet(model,train_loader,val_loader,lr=1e-03,Epochs = 20)

#test on some images
randomlist = random.sample(range(10, 30), 5)
model = model.cpu()
for n in randomlist:
  data = train_data[n]
  img = np.transpose(data[0].numpy(), (1, 2, 0))
  img = img[:,:,0] #for gray scale only "D array needed"
  plt.imshow(img,cmap='gray')

  with torch.no_grad():
    x = torch.tensor(img)
    x = x.float()
    _,_,pred_class = model(x.view(-1,model.in_size))

  plt.show()
  print("Predicted Value",pred_class )

"""**Cobvolutional networks**"""

out_size = 10 #10 classes form 0 to 9

img_size = (img.shape[0],img.shape[1])

Conv_model = ConvNetClassifier(img_size,5,out_size,in_channels=1,out_channels=1)
Conv_model.to(cuda0)

TrainNet(Conv_model,train_loader,val_loader,lr=1e-03,Epochs = 20,ConvNet=True)

#test on some images
randomlist = random.sample(range(10, 30), 5)
Conv_model = Conv_model.cpu()
for n in randomlist:
  data = train_data[n]
  img = np.transpose(data[0].numpy(), (1, 2, 0))
  img = img[:,:,0] #for gray scale only "D array needed"
  plt.imshow(img,cmap='gray')

  with torch.no_grad():
    img = data[0].numpy()
    img_input = torch.empty((1,img.shape[0],img.shape[1],img.shape[2]))
    img_input[0,:,:,:] = torch.tensor(img)
    x = torch.tensor(img_input)
    x = x.float()
    _,proba,pred_class = Conv_model(x)
    
  plt.show()
  print("Predicted Value",pred_class.numpy() )
  print("Predicted proba",proba.numpy() )