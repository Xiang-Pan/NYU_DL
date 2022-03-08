'''
Author: Xiang Pan
Date: 2022-03-05 01:56:18
LastEditTime: 2022-03-05 01:56:52
LastEditors: Xiang Pan
Description: 
FilePath: /HW3/hw3.py
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