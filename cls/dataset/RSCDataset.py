import os
import cv2
import time
import copy
import torch
import random
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pickle
from PIL import Image
from tqdm import tqdm
from glob import glob
from torch.utils.data import Dataset,DataLoader

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import albumentations as A
from albumentations.pytorch import ToTensorV2


class RSCDataset(Dataset):
    def __init__(self, datasets, transform=None):
        self.datasets = datasets
        self.transform = transform

    def __len__(self):
        return len(self.datasets)

    @classmethod
    def preprocess(cls, pil_img):
        img_nd = np.array(pil_img)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        try:
            img_trans = img_nd.transpose(2, 0, 1)
        except:
            print(img_nd.shape)
        if img_trans.max() > 1: img_trans = img_trans / 255
        return img_trans

    def __getitem__(self, index):
        dataset = self.datasets[index]
        file_path, label = dataset

        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        
        transformed = self.transform(image=image)
        image = transformed['image']
        
        return {
            'image': image,
            'label': label,
            "file_path": file_path
        }

if __name__ == '__main__':
    '''
    train_imgs_dir = "train_datasets/"
    with open('labels.pkl', 'rb') as f:
        labels = pickle.load(f)

    train_data = RSCDataset(train_imgs_dir, labels, transform=train_transform)
    train_loader = DataLoader(dataset=train_data, 
                                batch_size=8, 
                                shuffle=True,
                                num_workers=8)
    for batch_idx, batch_samples in enumerate(train_loader):
        image, target = batch_samples['image'], batch_samples['label']
        print(image.shape, target)

        if batch_idx > 10:
            break
    '''
    with open('datasets_pkl', 'rb') as f:
        datasets_pkl = pickle.load(f)

    train_data = RSCDataset(datasets_pkl["train"]+datasets_pkl["boost_train_dir"], transform=train_transform)
    train_loader = DataLoader(dataset=train_data, 
                                batch_size=24, 
                                shuffle=False,
                                num_workers=8)
    for batch_idx, batch_samples in tqdm(enumerate(train_loader)):
        image = batch_samples['image']
        print(image.shape)



