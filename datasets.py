import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import cv2
import os
import random
import albumentations as albu
import random
import math

class ManiDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, w=512, h=512, mode="train", resize=True, crop=False):
        # print(root, split)
        assert len(root) == len(split)
        
        self.splits = [] 
        for i, s in enumerate(split):
            self.splits.append(os.path.join(root[i], s))
            
        self.roots = root
        self.imgs = []
        self.labels = []
        self.w = w
        self.h = h
        
        self.mode = mode
        self.resize = resize
        self.crop = crop
        
        self.setup(self.roots, self.splits)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        if self.mode == 'train' or self.mode == 'semi-train':
            self.albu = albu.Compose([
                albu.RandomBrightnessContrast(p=0.5),
                # albu.HorizontalFlip(p=0.5),
                # albu.OneOf([
                #     albu.VerticalFlip(p=0.5),
                #     albu.ShiftScaleRotate(p=0.2, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
                # ]),
                # albu.OneOf([
                #     albu.ImageCompression(quality_lower=20, quality_upper=50, p=0.5),
                #     albu.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=False, p=0.5),
                #     albu.GaussNoise(var_limit=(100, 150), p=0.5)
                # ], p=0.5),                
                albu.OneOf([
                    albu.Rotate(limit=[90,90], p=0.5),
                    albu.Rotate(limit=[270,270], p=0.5),
                ], p=0.5),
                # albu.RandomCrop(self.h, self.w, p=1),
                albu.RandomResizedCrop(self.h, self.w, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), p=1),
                albu.Resize(self.h, self.w, p=1),
            ])
            # self.albu = albu.Compose([
            #     albu.HorizontalFlip(p=0.5),
            #     albu.OneOf([
            #         albu.VerticalFlip(p=0.5),
            #         albu.RandomRotate90(p=0.5),
            #         albu.RandomBrightnessContrast(p=0.2, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
            #         albu.HueSaturationValue(p=0.2, hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2),
            #         albu.ShiftScaleRotate(p=0.2, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
            #         albu.CoarseDropout(p=0.2),
            #         albu.Transpose(p=0.5)
            #     ]),
            #     albu.Resize(self.h, self.w, p=1),
            # ])
        elif self.mode == 'val':
            self.albu = albu.Compose([
                albu.Resize(self.h, self.w, p=1),
            ])
    
    def setup(self, roots, splits):
        for i, split in enumerate(splits):
            root = roots[i]
            with open(split, 'r') as f:
                while True:
                    line = f.readline().strip("\n")
                    if line:
                        self.imgs.append(os.path.join(root, "imgs/", line+".jpg"))
                        if self.mode == 'train' or self.mode == 'val' or self.mode == 'val-rawsize':
                            self.labels.append(os.path.join(root, "masks/", line+".png"))
                    else:
                        break
        # print(self.imgs)
        
    def __getitem__(self, index):
        # print(self.imgs[index])
        fake = cv2.imread(self.imgs[index])
        fake = cv2.cvtColor(fake, cv2.COLOR_BGR2RGB)
        fake_size = fake.shape
        
        if self.mode == 'train' or self.mode == 'val':
            label = cv2.imread(self.labels[index], cv2.IMREAD_GRAYSCALE)
            if(fake_size[0] < self.h or fake_size[1] < self.w):
                fake = cv2.resize(fake, (self.h,self.w))
                label = cv2.resize(label, (self.h,self.w)) 
            else:
                augmented = self.albu(image=fake, mask=label)
                fake, label = augmented['image'], augmented['mask']
            _, label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)           
            
            img = self.transform(fake)
            label = self.mask_transform(label)
            return img.float(), label.float()
        
        elif self.mode == "val-rawsize":
            label = cv2.imread(self.labels[index], cv2.IMREAD_GRAYSCALE)
            _, label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)           
            
            # if(fake_size[0] < self.h or fake_size[1] < self.w):
            #     fake = cv2.resize(fake, (self.h,self.w))
            #     label = cv2.resize(label, (self.h,self.w)) 

            h, w, c = fake_size
            window_size = self.h
            if h<window_size or w<window_size:
                size = min(h,w)
                ratio = float(window_size) / float(size)
                if h==w:
                    fake = cv2.resize(fake, (window_size, window_size))
                    label = cv2.resize(label,(window_size, window_size))
                elif size == h:
                    fake = cv2.resize(fake, (window_size, int(w*ratio)))
                    label = cv2.resize(label,(window_size, int(w*ratio)))
                else:
                    fake = cv2.resize(fake, (int(h*ratio), window_size))
                    label = cv2.resize(label, (int(h*ratio), window_size))

            img = self.transform(fake)
            label = self.mask_transform(label)
            return img.float(), label.float()
        
        elif self.mode == 'test':
            if self.resize:
                fake = cv2.resize(fake, (self.h, self.w))
            else:
                h, w, c = fake_size
                window_size = self.h
                if h<window_size or w<window_size:
                    size = min(h,w)
                    ratio = float(window_size) / float(size)
                    if h==w:
                        fake = cv2.resize(fake, (window_size, window_size))
                        # label = cv2.resize(label,(window_size, window_size))
                    elif size == h:
                        fake = cv2.resize(fake, (window_size, int(w*ratio)))
                        # label = cv2.resize(label,(window_size, int(w*ratio)))
                    else:
                        fake = cv2.resize(fake, (int(h*ratio), window_size))
                        # label = cv2.resize(label, (int(h*ratio), window_size))
            img = self.transform(fake)
            img_name = self.imgs[index].split("/")[-1].split('.')[0]
            return img.float(), img_name, fake_size
            
        elif self.mode == 'semi-train':
            augmented = self.albu(image=fake)
            fake = augmented['image']
            img = self.transform(fake)
            return img.float()
        
    def __len__(self):
        return len(self.imgs)


class ManiDataset2Task(torch.utils.data.Dataset):
    def __init__(self, root, split, w=512, h=512, mode="train", resize=True, crop=False):
        # print(root, split)
        assert len(root) == len(split)
        
        self.splits = [] 
        for i, s in enumerate(split):
            self.splits.append(os.path.join(root[i], s))
            
        self.roots = root
        self.imgs = []
        self.labels = []
        self.w = w
        self.h = h
        
        self.mode = mode
        self.resize = resize
        self.crop = crop
        
        self.setup(self.roots, self.splits)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        if self.mode == 'train' or self.mode == 'semi-train':
            self.albu = albu.Compose([
                albu.RandomBrightnessContrast(p=0.5),
                # albu.HorizontalFlip(p=0.5),
                # albu.OneOf([
                #     albu.VerticalFlip(p=0.5),
                #     albu.ShiftScaleRotate(p=0.2, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
                # ]),
                # albu.OneOf([
                #     albu.ImageCompression(quality_lower=20, quality_upper=50, p=0.5),
                #     albu.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=False, p=0.5),
                #     albu.GaussNoise(var_limit=(100, 150), p=0.5)
                # ], p=0.5),                
                albu.OneOf([
                    albu.Rotate(limit=[90,90], p=0.5),
                    albu.Rotate(limit=[270,270], p=0.5),
                ], p=0.5),
                # albu.RandomCrop(self.h, self.w, p=1),
                albu.RandomResizedCrop(self.h, self.w, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), p=1),
                albu.Resize(self.h, self.w, p=1),
            ])
        elif self.mode == 'val':
            self.albu = albu.Compose([
                albu.Resize(self.h, self.w, p=1),
            ])
    
    def setup(self, roots, splits):
        for i, split in enumerate(splits):
            root = roots[i]
            with open(split, 'r') as f:
                while True:
                    line = f.readline().strip("\n")
                    if line:
                        self.imgs.append(os.path.join(root, "imgs/", line+".jpg"))
                        if self.mode == 'train' or self.mode == 'val' or self.mode == 'val-rawsize':
                            self.labels.append(os.path.join(root, "masks/", line+".png"))
                    else:
                        break
        # print(self.imgs)
        
    def __getitem__(self, index):
        # print(self.imgs[index])
        fake = cv2.imread(self.imgs[index])
        fake = cv2.cvtColor(fake, cv2.COLOR_BGR2RGB)
        fake_size = fake.shape
        
        if self.mode == 'train' or self.mode == 'val':
            label = cv2.imread(self.labels[index], cv2.IMREAD_GRAYSCALE)
            if(fake_size[0] < self.h or fake_size[1] < self.w):
                fake = cv2.resize(fake, (self.h,self.w))
                label = cv2.resize(label, (self.h,self.w)) 
            else:
                augmented = self.albu(image=fake, mask=label)
                fake, label = augmented['image'], augmented['mask']
            _, label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)           
            
            if label.max() == 255:
                cls_label = torch.tensor(1).long()
            else:
                cls_label = torch.tensor(0).long()

            img = self.transform(fake)
            label = self.mask_transform(label)
            return img.float(), label.float(), cls_label
        
        elif self.mode == "val-rawsize":
            label = cv2.imread(self.labels[index], cv2.IMREAD_GRAYSCALE)
            _, label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)           
            
            img = self.transform(fake)
            label = self.mask_transform(label)

            if label.max() == 1:
                cls_label = torch.tensor(1).long()
            else:
                cls_label = torch.tensor(0).long()

            return img.float(), label.float(), cls_label
        
        elif self.mode == 'test':
            if self.resize:
                fake = cv2.resize(fake, (self.h, self.w))
            img = self.transform(fake)
            img_name = self.imgs[index].split("/")[-1].split('.')[0]
            return img.float(), img_name, fake_size
        
        # elif self.mode == 'semi-train':
        #     augmented = self.albu(image=fake)
        #     fake = augmented['image']
        #     img = self.transform(fake)
        #     return img.float()
        
    def __len__(self):
        return len(self.imgs)


def cal_new_mask(new_img, img, mask):
    """
    new img: 二次篡改的图片
    img：原来训练集中的图片
    mask：二次篡改前的标签，0-255
    """
    diff_img = cv2.absdiff(new_img, img)
    diff = np.linalg.norm(diff_img, ord=np.inf, axis=2)
    # print(diff.shape, mask.shape)
    _, diff = cv2.threshold(diff, 1, 255, cv2.THRESH_BINARY)
    
    new_mask = diff + mask
    new_mask = np.clip(new_mask, 0, 255)
    
    return new_mask

def rand_bbox(size):
    # opencv格式的size
    W = size[1]
    H = size[0]
        
    cut_rat_w = random.random()*0.1 + 0.05
    cut_rat_h = random.random()*0.1 + 0.05

    cut_w = int(W * cut_rat_w)
    cut_h = int(H * cut_rat_h)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W) # 左上
    bby1 = np.clip(cy - cut_h // 2, 0, H) # 左上
    bbx2 = np.clip(cx + cut_w // 2, 0, W) # 右下
    bby2 = np.clip(cy + cut_h // 2, 0, H) # 右下

    return bbx1, bby1, bbx2, bby2

def copy_move(img1, img2, msk, is_plot=False):
    img = img1.copy()
    size = img.shape # h,w,c
    W = size[1]
    H = size[0]

    if img2 is None: # 从自身复制粘贴
        bbx1, bby1, bbx2, bby2 = rand_bbox(img.shape)

        x_move = random.randrange(-bbx1, (W - bbx2))
        y_move = random.randrange(-bby1, (H - bby2))

        img[bby1+y_move:bby2+y_move, bbx1+x_move:bbx2+x_move, :] = img[bby1:bby2, bbx1:bbx2, :]
        
    else: # 从其他图像复制粘贴
        bbx1, bby1, bbx2, bby2 = rand_bbox(img2.shape)

        x_move = random.randrange(-bbx1, (W - bbx2))
        y_move = random.randrange(-bby1, (H - bby2))

        img[bby1+y_move:bby2+y_move, bbx1+x_move:bbx2+x_move, :] = img2[bby1:bby2, bbx1:bbx2, :]

    """ 
    这里改了一下dave的代码中直接根据修改区域计算mask，因为我发现有时候裁剪了一样的区域粘贴过来，
    计算方法是二次篡改的图片减去原图，有差异的地方叠加到原来的mask上
    """
    msk = cal_new_mask(img, img1, msk)
    if is_plot: # 标出二次窜改的区域，主要是为了debug，生成图像的时候记得改成false
        img =  cv2.rectangle(img, pt1=[bbx1+x_move, bby1+y_move], pt2=[bbx2+x_move, bby2+y_move], color=(255,0,0), thickness=3)       

    return np.uint8(img), np.uint8(msk)

def erase(img1, msk, is_plot=False):
    img = img1.copy()
    size = img.shape # h,w,c
    W = size[1]
    H = size[0]

    def midpoint(x1, y1, x2, y2):
        x_mid = int((x1 + x2)/2)
        y_mid = int((y1 + y2)/2)
        return (x_mid, y_mid)

    bbx1, bby1, bbx2, bby2 = rand_bbox(img.shape)
    # print(bbx1, bby1, bbx2, bby2)
    
    x_mid0, y_mid0 = midpoint(bbx1, bby1, bbx1, bby2)
    x_mid1, y_mid1 = midpoint(bbx2, bby1, bbx2, bby2)
    thickness = int(math.sqrt((bby2-bby1)**2))
    
    mask_ = np.zeros(img.shape[:2], dtype="uint8")    
    cv2.line(mask_, (x_mid0, y_mid0), (x_mid1, y_mid1), 255, thickness)
    
    # cv2.imwrite("mask_.jpg", mask_)
    img = cv2.inpaint(img, mask_, 7, cv2.INPAINT_NS)
    
    msk = cal_new_mask(img1, img, msk)
    
    if is_plot:             
        img = cv2.rectangle(img, pt1=[bbx1, bby1], pt2=[bbx2, bby2], color=(255,0,0), thickness=3)
        
    return np.uint8(img), np.uint8(msk)

def mosaic_effect(img):
    img = img.numpy().transpose(1,2,0) 
    h, w, n = img.shape
    # size = random.randint(5, 20) #马赛克大小
    size = 9
    for i in range(0, h - size, size):
        for j in range(0, w - size, size):
            rect = [j, i, size, size]
            color = img[i, j].tolist()
            left_up = (rect[0], rect[1])
            right_down = (rect[0]+size, rect[1]+size)
            cv2.rectangle(img, left_up, right_down, color, -1)    
    return torch.from_numpy(img).permute(2, 0, 1)
    
def mosaic(img, msk):
    resize = albu.Resize(512,512)(image=img, mask=msk)
    img = torch.from_numpy(resize['image']).permute(2, 0, 1)
    msk = torch.from_numpy(resize['mask'])
    size = img.size()
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    bbx1, bby1, bbx2, bby2 = rand_bbox(img.size())

    x_move = random.randrange(-bbx1, (W - bbx2))
    y_move = random.randrange(-bby1, (H - bby2))

    img[:, bbx1+x_move:bbx2+x_move, bby1+y_move:bby2+y_move] = mosaic_effect(img[:, bbx1:bbx2, bby1:bby2])
    msk[bbx1+x_move:bbx2+x_move, bby1+y_move:bby2+y_move] = torch.ones_like(msk[bbx1:bbx2, bby1:bby2])*255 
    
    # img = img.numpy().transpose(1,2,0)
    img = cv2.rectangle(img.numpy().transpose(1,2,0),pt1=[bby1+y_move, bbx1+x_move], pt2=[bby2+y_move, bbx2+x_move], color=(255,0,0), thickness=3)
    msk = msk.numpy()  
    
    return img, msk

class ManiDatasetAug(torch.utils.data.Dataset):
    def __init__(self, root, split, w=512, h=512, mode="train", resize=True, crop=False):
        # print(root, split)
        assert len(root) == len(split)
        
        self.splits = [] 
        for i, s in enumerate(split):
            self.splits.append(os.path.join(root[i], s))
            
        self.roots = root
        self.imgs = []
        self.labels = []
        self.w = w
        self.h = h
        
        self.mode = mode
        self.resize = resize
        self.crop = crop
        
        self.setup(self.roots, self.splits)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        if self.mode == 'train' or self.mode == 'semi-train':
            self.albu = albu.Compose([
                albu.RandomBrightnessContrast(p=0.5),
                # albu.OneOf([
                #     albu.ImageCompression(quality_lower=20, quality_upper=50, p=0.5),
                #     albu.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=False, p=0.5),
                #     albu.GaussNoise(var_limit=(100, 150), p=0.5)
                # ], p=0.5),                
                albu.OneOf([
                    albu.Rotate(limit=[90,90], p=0.5),
                    albu.Rotate(limit=[270,270], p=0.5),
                ], p=0.5),
                # albu.RandomCrop(self.h, self.w, p=1),
                albu.RandomResizedCrop(self.h, self.w, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), p=1),
                albu.Resize(self.h, self.w, p=1),
            ])
        elif self.mode == 'val':
            self.albu = albu.Compose([
                albu.Resize(self.h, self.w, p=1),
            ])
    
    def setup(self, roots, splits):
        for i, split in enumerate(splits):
            root = roots[i]
            with open(split, 'r') as f:
                while True:
                    line = f.readline().strip("\n")
                    if line:
                        self.imgs.append(os.path.join(root, "imgs/", line+".jpg"))
                        if self.mode == 'train' or self.mode == 'val' or self.mode == 'val-rawsize':
                            self.labels.append(os.path.join(root, "masks/", line+".png"))
                    else:
                        break
        # print(self.imgs)
        
    def __getitem__(self, index):
        fake = cv2.imread(self.imgs[index])
        fake = cv2.cvtColor(fake, cv2.COLOR_BGR2RGB)
        fake_size = fake.shape
        
        if self.mode == 'train':
            label = cv2.imread(self.labels[index], cv2.IMREAD_GRAYSCALE)
            
            p = random.randint(0,3)
            # print(p)
            if p == 2:
                #自身随机裁切
                fake, label = copy_move(fake, None, label)
            elif p == 3:
                # 从其他图片随机裁切
                img2 = cv2.imread(self.imgs[random.randint(0, len(self.imgs)-1)])
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                fake, label = copy_move(fake, img2, label)
            elif p == 1: 
                # 随机擦除
                fake, label = erase(fake, label)
            
            # # 随机马赛克:
            # # fake, label = mosaic(fake, label)
            
            if(fake_size[0] < self.h or fake_size[1] < self.w):
                fake = cv2.resize(fake, (self.h,self.w))
                label = cv2.resize(label, (self.h,self.w)) 
            else:
                augmented = self.albu(image=fake, mask=label)
                fake, label = augmented['image'], augmented['mask']
            _, label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)
            
            # cv2.imwrite("aug_img.jpg", fake)
            # cv2.imwrite("aug_mask.jpg", label)           
            
            img = self.transform(fake)
            label = self.mask_transform(label)       
            
            return img.float(), label.float()
        
        elif self.mode == 'val':
            label = cv2.imread(self.labels[index], cv2.IMREAD_GRAYSCALE)
            if(fake_size[0] < self.h or fake_size[1] < self.w):
                fake = cv2.resize(fake, (self.h,self.w))
                label = cv2.resize(label, (self.h,self.w)) 
            else:
                augmented = self.albu(image=fake, mask=label)
                fake, label = augmented['image'], augmented['mask']
            _, label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)           
            
            img = self.transform(fake)
            label = self.mask_transform(label)
            return img.float(), label.float()
        
        elif self.mode == "val-rawsize":
            label = cv2.imread(self.labels[index], cv2.IMREAD_GRAYSCALE)
            _, label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)           
            
            img = self.transform(fake)
            label = self.mask_transform(label)
            return img.float(), label.float()
        
        elif self.mode == 'test':
            if self.resize:
                fake = cv2.resize(fake, (self.h, self.w))
            img = self.transform(fake)
            img_name = self.imgs[index].split("/")[-1].split('.')[0]
            return img.float(), img_name, fake_size
        
    def __len__(self):
        return len(self.imgs)
    
if __name__ == "__main__":
    """ 测试dataset类 """
    root = "/data1/datasets/ICDAR2023/train/tampered"
    trainset = ManiDataset([root], split=["train_pos.txt"], w=512, h=512, )
    trainloader = data.DataLoader(trainset, batch_size=16, shuffle=True)
    
    for img, mask in trainloader:
        print(img.shape, mask.shape)
        print(mask)
        print(mask.max())
        break