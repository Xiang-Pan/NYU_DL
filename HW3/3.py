'''
Author: Xiang Pan
Date: 2022-03-05 01:56:18
LastEditTime: 2022-03-08 18:44:41
LastEditors: Xiang Pan
Description: 
FilePath: /HW3/3.py
@email: xiangpan@nyu.edu
'''
from PIL import ImageDraw, ImageFont
import string
import random
import torch
import torchvision
from torchvision import transforms
from PIL import Image # PIL is a library to process images
from matplotlib import pyplot as plt

ALPHABET_SIZE = 26

simple_transforms = transforms.Compose([
                                    transforms.ToTensor(), 
                                ])

class SimpleWordsDataset(torch.utils.data.IterableDataset):

    def __init__(self, max_length, len=100, jitter=False, noise=False):
        self.max_length = max_length
        self.transforms = transforms.ToTensor()
        self.len = len
        self.jitter = jitter
        self.noise = noise

    def __len__(self):
        return self.len

    def __iter__(self):
        for _ in range(self.len):
            text = ''.join([random.choice(string.ascii_lowercase) for i in range(self.max_length)])
            img = self.draw_text(text, jitter=self.jitter, noise=self.noise)
            yield img, text

    def draw_text(self, text, length=None, jitter=False, noise=False):
        if length == None:
            length = 18 * len(text)
        img = Image.new('L', (length, 32))
        fnt = ImageFont.truetype("fonts/Anonymous.ttf", 20)

        d = ImageDraw.Draw(img)
        pos = (0, 5)
        if jitter:
            pos = (random.randint(0, 7), 5)
        else:
            pos = (0, 5)
        d.text(pos, text, fill=1, font=fnt)

        img = self.transforms(img)
        img[img > 0] = 1 

        if noise:
            img += torch.bernoulli(torch.ones_like(img) * 0.1)
            img = img.clamp(0, 1)
            

        return img[0]

# sds = SimpleWordsDataset(1, jitter=True, noise=False)
# img = next(iter(sds))[0]
# print(img.shape)
# plt.imshow(img)

'''
Author: Xiang Pan
Date: 2022-03-05 01:56:18
LastEditTime: 2022-03-08 18:22:08
LastEditors: Xiang Pan
Description: 
FilePath: /HW3/hw3_practice.ipynb
@email: xiangpan@nyu.edu
'''
from PIL import ImageDraw, ImageFont
import string
import random
import torch
import torchvision
from torchvision import transforms
from PIL import Image # PIL is a library to process images
from matplotlib import pyplot as plt

simple_transforms = transforms.Compose([
                                    transforms.ToTensor(), 
                                ])

import torch.nn as nn
# sliding window cnn
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.filters = [8, 16, 32]
        self.width = 18 
        self.cnn_block = []
        self.cnn_block = torch.nn.Sequential(
            # TODO
            torch.nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(32, ALPHABET_SIZE, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        batch_size = x.shape[0]

        logit = self.cnn_block(x)
        logit = logit.view(batch_size, 1, ALPHABET_SIZE, -1)
        return logit


class SimpleWordsDataset(torch.utils.data.IterableDataset):

    def __init__(self, max_length, len=100, jitter=False, noise=False):
        self.max_length = max_length
        self.transforms = transforms.ToTensor()
        self.len = len
        self.jitter = jitter
        self.noise = noise

    def __len__(self):
        return self.len

    def __iter__(self):
        for _ in range(self.len):
            text = ''.join([random.choice(string.ascii_lowercase) for i in range(self.max_length)])
            img = self.draw_text(text, jitter=self.jitter, noise=self.noise)
            yield img, text

    def draw_text(self, text, length=None, jitter=False, noise=False):
        if length == None:
            length = 18 * len(text)
        img = Image.new('L', (length, 32))
        fnt = ImageFont.truetype("fonts/Anonymous.ttf", 20)

        d = ImageDraw.Draw(img)
        pos = (0, 5)
        if jitter:
            pos = (random.randint(0, 7), 5)
        else:
            pos = (0, 5)
        d.text(pos, text, fill=1, font=fnt)

        img = self.transforms(img)
        img[img > 0] = 1 

        if noise:
            img += torch.bernoulli(torch.ones_like(img) * 0.1)
            img = img.clamp(0, 1)
            

        return img[0]

sds = SimpleWordsDataset(1, jitter=True, noise=False)
img = next(iter(sds))[0]
print(img.shape)
plt.imshow(img)


def get_accuracy(model, dataset):
    cnt = 0
    for i, l in dataset:
        # batch_size, 1, alphabet_size, width
        energies = model(i.unsqueeze(0).unsqueeze(0).cuda())[0, 0]
        x = energies.argmin(dim=-1)
        cnt += int(x == (ord(l[0]) - ord('a')))
    return cnt / len(dataset)
        

# assert get_accuracy(model, tds) == 1.0, 'Your model doesn\'t achieve 100% accuracy for 1 character'

def train_model(model, epochs, dataloader, criterion, optimizer):
    for epoch in range(epochs):
        for i, (img, text) in enumerate(dataloader):
            img = img.cuda()
            img = img.unsqueeze(1)
            text = text.cuda()
            optimizer.zero_grad()
            energy = model(img)
            print(energy.shape)
            # energy = torch.sum(output, dim=2)
            # get acc
            # tds = SimpleWordsDataset(1, len=100)
            # print(get_accuracy(model, tds))
            acc = torch.sum(energy.argmin(dim=2) == text) / len(text)
            loss = criterion(energy, text)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'Epoch {epoch}, iteration {i}, loss {loss.item():.4f}, acc {acc.item():.4f}')
            
        

from tqdm.notebook import tqdm
import torch.optim as optim

def cross_entropy(energies, *args, **kwargs):
    """ We use energies, and therefore we need to use log soft arg min instead
        of log soft arg max. To do that we just multiply energies by -1. """
    return nn.functional.cross_entropy(-1 * energies, *args, **kwargs)

def simple_collate_fn(samples):
    images, annotations = zip(*samples)
    images = list(images)
    annotations = list(annotations)
    annotations = list(map(lambda c : torch.tensor(ord(c) - ord('a')), annotations))
    m_width = max(18, max([i.shape[1] for i in images]))
    for i in range(len(images)):
        images[i] = torch.nn.functional.pad(images[i], (0, m_width - images[i].shape[-1]))
        
    if len(images) == 1:
        return images[0].unsqueeze(0), torch.stack(annotations)
    else:
        return torch.stack(images), torch.stack(annotations)

sds = SimpleWordsDataset(1, len=1000, jitter=True, noise=False)
dataloader = torch.utils.data.DataLoader(sds, batch_size=16, num_workers=0, collate_fn=simple_collate_fn)

model = SimpleNet().cuda()
train_model(model, epochs=10, dataloader=dataloader, criterion=cross_entropy, optimizer=optim.Adam(model.parameters(), lr=0.001))