# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import time
from io import BytesIO
import base64
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000000000
from tqdm import tqdm
import glob
import os
from tqdm import tqdm
from scipy.io import loadmat
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
from dataset.RSCDataset import RSCDataset
from torch.utils.data import Dataset, DataLoader
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")



transform_512 = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
])

transform_768 = A.Compose([
    A.Resize(768, 768),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
])


# 网络
class seg_qyl(nn.Module):
    def __init__(self, model_name, n_class):
        super().__init__()  
        self.model = smp.UnetPlusPlus (# UnetPlusPlus / DeepLabV3Plus
                encoder_name=model_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
                classes=3,                      # model output channels (number of classes in your dataset)
                aux_params={"classes":768}
            )
        self.linear = nn.Linear(768, n_class)
    def forward(self, x):
        mask, out = self.model(x)
        out = self.linear(out)
        return out


def infer_function(val_transform, save_name, batch_size, checkpoint_dir):
    with open('dataset/datasets_pkl', 'rb') as f:
        datasets_pkl = pickle.load(f)

    test_data = RSCDataset(datasets_pkl["test"], transform=val_transform)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=8)


    model_name = 'efficientnet-b6'#
    n_class = 2
    model = seg_qyl(model_name, n_class)
    model = torch.nn.DataParallel(model)
    model.to(device)

    print(checkpoint_dir)

    checkpoints = torch.load(checkpoint_dir)
    if 'state_dict' in checkpoints.keys():
        print("loading state_dict")
        model.load_state_dict(checkpoints['state_dict'])
    else:
        model.load_state_dict(checkpoints)
    
    file_names = []
    preds = []
    model.eval()
    with torch.no_grad():
        for batch_idx, batch_samples in tqdm(enumerate(test_loader)):
            data, file_name = batch_samples['image'], batch_samples['label']
            data = Variable(data.to(device))
            pred = model(data)
            pred = F.softmax(pred, dim=1)
            preds += list(pred[:,1].cpu().numpy())
            file_names += list(file_name)
        
    with open(save_name,"w",encoding="utf-8") as f:
        for file_name, pred in zip(file_names, preds):
            f.writelines(file_name+"\t"+str(pred)+"\n")

    
if __name__=="__main__":
    infer_function(transform_512, "512.csv", 8, 'models/512.pth')
    infer_function(transform_768, "768.csv", 4, 'models/768.pth')