import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from Dataset import LandmarkTrainDataset , LandmarkTestDataset
import os

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_dataset = LandmarkTrainDataset(csv_file="train.csv", root_dir="train", transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True )


test_ids = [f.split('.')[0] for f in os.listdir('test') if f.endswith('.jpg')]
test_dataset = LandmarkTestDataset(test_ids, root_dir="test", transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)




def imshow(inp, title=None):

    inp = inp.numpy().transpose((1, 2, 0))  # (C,H,W) -> (H,W,C)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean  # denormalizăm imaginea
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title:
        plt.title(title)
    plt.pause(0.001)



inputs, classes = next(iter(train_loader))


out = torchvision.utils.make_grid(inputs[:4])
imshow(out, title=[str(c.item()) for c in classes[:4]])


