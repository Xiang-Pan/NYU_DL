'''
Author: Xiang Pan
Date: 2022-03-07 18:40:14
LastEditTime: 2022-03-07 20:43:00
LastEditors: Xiang Pan
Description: 
FilePath: /HW2/test.py
@email: xiangpan@nyu.edu
'''
import torch
import numpy as np
import torch.nn as nn
import copy
# cnn with residual link
import torch.nn.functional as F

# class Block(nn.Module):
#     def __init__(self,ni):
#         super(Block, self).__init__()
#         self.conv1 = nn.Conv2d(ni, ni, 1)
#         self.conv2 = nn.Conv2d(ni, ni, 3, 1, 1)
#         self.classifier = nn.Linear(ni*256*256,512)

#     def forward(self,x):
#         residual = x
#         out = F.relu(self.conv1(x))
#         out = F.relu(self.conv2(out))
        
#         out += residual
        
#         out = out.view(out.size(0),-1)
#         return self.classifier(out)

# class BaseCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.block1 = Block(3)
#         self.block2 = Block(3)
    
#     def forward(self,x):
#         x = self.block1(x)
#         print(x.shape)
#         x = self.block2(x)
#         return x




# model = BaseCNN()
# dummy_input = torch.randn(1, 3, 256, 256)
# print(model(dummy_input).shape)




class BaseCNN(nn.Module):
    def __init__(self):
        super(BaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AdaptiveAvgPool2d((128, 128))

        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.avgpool2 = nn.AdaptiveAvgPool2d((128, 128))
        
        self.conv3 = nn.Conv2d(32, 6, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(6)
        self.relu3 = nn.ReLU()
        self.avgpool2 = nn.AdaptiveAvgPool2d((128, 128))

        self.cls = nn.Linear(98304, 2)
    
    def forward(self, x):
        batch_size = x.size(0)
        t = self.conv1(x)
        t = self.bn1(t)
        t = self.relu1(t)
        t = self.avgpool1(t)

        r1 = t

        t = self.conv2(t)
        t = self.bn2(t)
        t = self.relu2(t)
        t += r1
        
        r2 = t
        t = self.conv3(t)
        t = self.bn3(t)
        t = self.relu3(t)
        
        final_rep = t.view(batch_size, -1)
        logit = self.cls(final_rep)

        return logit


model = BaseCNN()
dummy_input = torch.randn(1, 3, 256, 256)
print(model(dummy_input).shape)

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image # PIL is a library to process images

# # These numbers are mean and std values for channels of natural images. 
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])

# # Inverse transformation: needed for plotting.
# unnormalize = transforms.Normalize(
#     mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
#     std=[1/0.229, 1/0.224, 1/0.225]
# )

# train_transforms = transforms.Compose([
#                                     transforms.Resize((256, 256)),
#                                     transforms.RandomHorizontalFlip(),
#                                     transforms.ColorJitter(hue=.1, saturation=.1, contrast=.1),
#                                     transforms.RandomRotation(20, resample=Image.BILINEAR),
#                                     transforms.GaussianBlur(7, sigma=(0.1, 1.0)),
#                                     transforms.ToTensor(),  # convert PIL to Pytorch Tensor
#                                     normalize,
#                                 ])

# validation_transforms = transforms.Compose([
#                                     transforms.Resize((256, 256)),
#                                     transforms.ToTensor(), 
#                                     normalize,
#                                 ])

# train_dataset = torchvision.datasets.ImageFolder('./cats_and_dogs_filtered/train', transform=train_transforms)
# validation_dataset, test_dataset = torch.utils.data.random_split(torchvision.datasets.ImageFolder('./cats_and_dogs_filtered/validation', transform=validation_transforms), [500, 500], generator=torch.Generator().manual_seed(42))



# import torch
# from tqdm.notebook import tqdm

# def get_loss_and_correct(model, batch, criterion, device):
#     # Implement forward pass and loss calculation for one batch.
#     # Remember to move the batch to device.
#     # 
#     # Return a tuple:
#     # - loss for the batch (Tensor)
#     # - number of correctly classified examples in the batch (Tensor)
#     # print(batch)
#     x, y = batch
#     x = x.to(device)
#     y = y.to(device)
#     y_hat = model(x)
#     # print(y)
#     loss = criterion(y_hat, y)
#     correct = (torch.argmax(y_hat, dim=1) == y).sum()
#     return loss, correct

# def step(loss, optimizer):
#     # Implement backward pass and update.
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()





# N_EPOCHS = 20
# BATCH_SIZE = 64

# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
# validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers=4)
# model = BaseCNN()

# criterion = nn.CrossEntropyLoss()

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# model.train()

# if torch.cuda.is_available():
#     model = model.cuda()
#     criterion = criterion.cuda()
#     device = torch.device("cuda:0")
# else:
#     device = torch.device("cpu")

# train_losses = []
# train_accuracies = []
# validation_losses = []
# validation_accuracies = []

# pbar = tqdm(range(N_EPOCHS))

# for i in pbar:
#     total_train_loss = 0.0
#     total_train_correct = 0.0
#     total_validation_loss = 0.0
#     total_validation_correct = 0.0

#     model.train()

#     for batch in tqdm(train_dataloader, leave=False):
#         loss, correct = get_loss_and_correct(model, batch, criterion, device)
#         step(loss, optimizer)
#         total_train_loss += loss.item()
#         total_train_correct += correct.item()

#     with torch.no_grad():
#         for batch in validation_dataloader:
#             loss, correct = get_loss_and_correct(model, batch, criterion, device)
#             total_validation_loss += loss.item()
#             total_validation_correct += correct.item()

#     mean_train_loss = total_train_loss / len(train_dataset)
#     train_accuracy = total_train_correct / len(train_dataset)

#     mean_validation_loss = total_validation_loss / len(validation_dataset)
#     validation_accuracy = total_validation_correct / len(validation_dataset)

#     train_losses.append(mean_train_loss)
#     validation_losses.append(mean_validation_loss)

#     train_accuracies.append(train_accuracy)
#     validation_accuracies.append(validation_accuracy)

#     pbar.set_postfix({'train_loss': mean_train_loss, 'validation_loss': mean_validation_loss, 'train_accuracy': train_accuracy, 'validation_accuracy': validation_accuracy})


if batch_idx == len(train_data_gen)-1:
    # grad save
    model.linear_gradient_list.append(copy.deepcopy(model.linear.weight.grad))
    model.lstm_ih_gradient_list.append(copy.deepcopy(model.lstm.weight_ih_l0.grad))
    model.lstm_hh_gradient_list.append(copy.deepcopy(model.lstm.weight_hh_l0.grad))
 
    # weight save
    model.linear_weight_list.append(copy.deepcopy(model.linear.weight))
    model.lstm_ih_weight_list.append(copy.deepcopy(model.lstm.weight_ih_l0))
    model.lstm_hh_weight_list.append(copy.deepcopy(model.lstm.weight_hh_l0))

    # self.linear_gradient_list = []
    # self.lstm_ih_gradient_list = []
    # self.lstm_hh_gradient_list = []

    # self.linear_weight_list = []
    # self.lstm_ih_weight_list = []
    # self.lstm_hh_weight_list = []