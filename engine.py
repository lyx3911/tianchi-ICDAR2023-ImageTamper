from utils import *
from tqdm import tqdm
import torch.nn.functional as F
from losses import DiceLoss, SoftCrossEntropyLoss, LovaszLoss
import torchvision.transforms as transforms

def train_one_epoch(model, trainloader, optimizer):
    model.train()
    train_f1_score = AverageMeter() 
    train_ious = AverageMeter()
    ce = AverageMeter()
    dice = AverageMeter()

    ce_loss = SoftCrossEntropyLoss(smooth_factor=0.1)
    dice_loss = DiceLoss(mode="multiclass")

    iters = 0

    for images, labels in tqdm(trainloader):
        images, labels = images.cuda(), labels.cuda()
        seg = model(images)
        
        labels = labels.squeeze(1)
        
        # print(seg.shape, labels.shape)
        loss_ce = ce_loss(seg, labels.long())
        loss_dice = dice_loss(seg, labels.long())

        loss = 0.3*loss_ce + 0.7*loss_dice
        # loss = loss_ce
        # loss = loss_dice

        ce.update(loss_ce.item())
        dice.update(loss_dice.item())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2)
        optimizer.step()

        seg = F.softmax(seg, dim=1)[:,1]

        f1, iou = calculate_batch_score(seg, labels)
        train_f1_score.update(f1, images.shape[0])
        train_ious.update(iou, images.shape[0])
        
        iters += 1

        if iters >= 1000:
            break
        # break

    return ce.avg, dice.avg, train_f1_score.avg, train_ious.avg

def val_whole(model, valloader):
    """整张图像进行resize"""
    model.eval()
    f1_score = AverageMeter() 
    ious = AverageMeter()
    for images, labels in tqdm(valloader):
        images, labels =  images.cuda(), labels.cuda()
        with torch.no_grad():
            seg = model(images)
        # print(seg.shape, labels.shape)
        seg = F.softmax(seg, dim=1)[:,1]
        f1, iou = calculate_batch_score(seg, labels.squeeze(1))
        f1_score.update(f1, images.shape[0])
        ious.update(iou, images.shape[0])
    return f1_score.avg, ious.avg

def cal_sliding_params(img_h, img_w, window_size=512, overlap=128):
    # 计算需要裁剪成几块
    col, row = 1, 1
    while (window_size*col - (col-1)*overlap) < img_h:
        col += 1
    while (window_size*row - (row-1)*overlap) < img_w:
        row += 1
    return col, row

def img_slide_window(img, col, row, window_size=512):
    imgs = []
    # 计算 overlape
    delta_x, delta_y = 0, 0
    if row > 1:
        delta_x = int((row*window_size-img.shape[-1])/(row-1))
    if col > 1:
        delta_y = int((col*window_size-img.shape[-2])/(col-1))
        
    for i in range(col):
        for j in range(row):
            begin_h = window_size*i - max(0, i)*delta_y
            begin_w = window_size*j - max(0, j)*delta_x
            
            if begin_h + window_size > img.shape[-2]:
                begin_h = img.shape[-2] - window_size
            if begin_w + window_size > img.shape[-1]:
                begin_w = img.shape[-1] - window_size
            slide = img[:, :, begin_h:begin_h+window_size, begin_w:begin_w+window_size].squeeze(0)
            imgs.append(slide)
            # print(begin_h, begin_w, begin_h+512, begin_w+512, img.shape)
    return torch.stack(imgs, dim=0)

def merge_slides_result(segs, col, row, img_shape, window_size=512):
    count = torch.zeros([1, img_shape[2], img_shape[3]]).cuda()
    seg = torch.zeros([1, img_shape[2], img_shape[3]]).cuda()
    
    # 计算 overlape
    delta_x, delta_y = 0, 0
    if row > 1:
        delta_x = int((row*window_size-img_shape[-1])/(row-1))
    if col > 1:
        delta_y = int((col*window_size-img_shape[-2])/(col-1))
        
    # print(col, row)
    for i in range(col):
        for j in range(row):
            begin_h = window_size*i - max(0, i)*delta_y
            begin_w = window_size*j - max(0, j)*delta_x
            
            if begin_h + window_size > img_shape[-2]:
                begin_h = img_shape[-2] - window_size
            if begin_w + window_size > img_shape[-1]:
                begin_w = img_shape[-1] - window_size
            seg[:, begin_h:begin_h+window_size, begin_w:begin_w+window_size] += segs[i*row+j]
            count[:, begin_h:begin_h+window_size, begin_w:begin_w+window_size] += 1.0
    seg = seg / count
    return seg.unsqueeze(0)

def val_slide(model, valloader, window_size=768):
    scores = AverageMeter()
    ious = AverageMeter()
    model.eval()
    
    for img, label in tqdm(valloader):
        img, label = img.cuda(), label.cuda() #[3,h,w]
        # print(img.shape, label.shape)
        assert img.shape[0] == 1
        with torch.no_grad():
            img_h, img_w = img.shape[-2], img.shape[-1]
            col, row = cal_sliding_params(img_h, img_w, window_size=window_size, overlap=window_size/4)
            imgs = img_slide_window(img, col, row, window_size=window_size)
            # print(img_h, img_w, col, row, imgs.shape)
            # print(imgs.shape)
            seg = model(imgs)
            seg = F.softmax(seg, dim=1)[:,1].unsqueeze(1)
        # seg = torch.sigmoid(seg)
        seg = merge_slides_result(seg, col, row, img.shape, window_size=window_size)

        transform = transforms.Compose([
            transforms.Resize([window_size,window_size])
        ])
        invtransform = transforms.Compose([
            transforms.Resize([img.shape[-2], img.shape[-1]])
        ]) 

        # print("slide")
        with torch.no_grad():
            seg_resize = model(transform(img))
            seg_resize = invtransform(seg_resize)
            seg_resize = F.softmax(seg_resize, dim=1)[:,1].unsqueeze(1)

        seg = seg*0.7 + seg_resize*0.3
   
        f1, iou = calculate_batch_score(seg, label)
        scores.update(f1, 1)
        ious.update(iou, 1) 
    # print(losses.avg)    
    return scores.avg, ious.avg # return f1, iou