import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models.convnext import unet_convnext

# from config import get_config

import torch.optim as optim

from tqdm import tqdm
from datasets import ManiDataset, ManiDatasetAug
from torch.nn.modules.loss import CrossEntropyLoss
from utils import *
import torch.utils.data as data
import torch.nn.functional as F

from engine import *


parser = argparse.ArgumentParser()
# parser.add_argument('--root_path', type=str,
#                     default='../data/Synapse/train_npz', help='root dir for data')
# parser.add_argument('--dataset', type=str,
#                     default='Synapse', help='experiment_name')
# parser.add_argument('--list_dir', type=str,
#                     default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--output_dir', type=str, help='output dir')                   
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=42, help='random seed')
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()


if __name__ == "__main__":
    # 固定随机数种子
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model = unet_convnext("base", img_size=args.img_size, out_channels=2).cuda()

    from collections import OrderedDict
    state_dict = torch.load("model_save/unet-convnext-1024-0303-1e-5/best.pth")
    new_state_dict = OrderedDict()   # create new OrderedDict that does not contain `module.`
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=True)

    # 训练参数设置
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9,0.999), weight_decay=5e-4)
    
    # dataloader
    # root = ["/data1/datasets/ICDAR2023/train/tampered", "/data1/datasets/ICDAR2023/train/untampered"]
    root = ["/data1/datasets/ICDAR2023/train/tampered-train-slide-window-768",
    "/data1/datasets/ICDAR2023/train/tampered-train-slide-window-768",
    "/data1/datasets/ICDAR2023/train/tampered-train-slide-window-1024",
    "/data1/datasets/ICDAR2023/train/tampered-train-slide-window-1024",
    "/data1/datasets/ICDAR2023/train/tampered",
    "/data1/datasets/ICDAR2023/train/untampered",
    ]
    rootval = ["/data1/datasets/ICDAR2023/train/tampered"]
    trainset = ManiDataset(root, split=["train_pos.txt", "train_neg.txt", "train_pos.txt", "train_neg.txt", "train_pos.txt", "train_neg.txt",], 
    w=args.img_size, h=args.img_size, mode='train')
    trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    valset = ManiDataset(rootval, split=["val_pos.txt", ], w=args.img_size, h=args.img_size, mode='val-rawsize')
    valloader = data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=4)

    best_iou = 0.0

    for epoch in range(args.max_epochs):
        ce, dice, train_f1_score, train_iou_score = train_one_epoch(model, trainloader, optimizer)
        print("epoch", epoch, "ce loss: ", ce, "dice loss", dice)

        f1, iou = val_slide(model, valloader, window_size=args.img_size)        
        print("train f1:", train_f1_score, "iou:", train_iou_score, "// test f1:", f1, "iou:", iou)

        if iou > best_iou:
            torch.save(model.state_dict(), "{}/best.pth".format(args.output_dir))
            best_iou = iou
        torch.save(model.state_dict(), "{}/curr.pth".format(args.output_dir))