import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models.convnext import unet_convnext, unetplusplus_convnext
import segmentation_models_pytorch as smp
from collections import OrderedDict

# from config import get_config

import torch.optim as optim

from tqdm import tqdm
from datasets import ManiDataset, ManiDatasetAug
from torch.nn.modules.loss import CrossEntropyLoss
from utils import *
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2

from losses import DiceLoss, SoftCrossEntropyLoss


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
# parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
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
parser.add_argument('--model_path', type=str,
                    default="./model_save/best.pth", help='input patch size of network input')

args = parser.parse_args()
# if args.dataset == "Synapse":
#     args.root_path = os.path.join(args.root_path, "train_npz")
# config = get_config(args)


def infer(model, img):
    transform_pil = transforms.Compose([
        transforms.ToPILImage(),
    ])
    img = img.cuda().view(-1, img.shape[0], img.shape[1], img.shape[2])

    transform = transforms.Compose([
        transforms.Resize([args.img_size, args.img_size])
    ])
    invtransform = transforms.Compose([
        transforms.Resize([img.shape[-2], img.shape[-1]])
    ]) 

    with torch.no_grad():
        seg = model(transform(img))
        seg = F.softmax(seg, dim=1)
        seg = invtransform(seg)[0,1].detach().cpu().numpy()
    
    # seg[seg<0.05]=0
    mask = seg * 255.0
    mask = mask.astype(np.uint8)
    
    return mask

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
    
    # print(config)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model = unet_convnext("base", img_size=args.img_size).cuda()

    
    state_dict = torch.load(args.model_path)
    new_state_dict = OrderedDict()   # create new OrderedDict that does not contain `module.`
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=True)
    
    # dataloader
    # testset = ManiDataset(["/data1/datasets/ICDAR2023/train/untampered/",], split=["val_neg.txt"], w=args.img_size, h=args.img_size, mode='test')
    testset = ManiDataset(["../tianchi_data/data/test/",], split=["test.txt"], w=args.img_size, h=args.img_size, mode='test', resize=True)
    print(len(testset))
    model.eval()
    for img, img_name, img_size in tqdm(testset):
        mask = infer(model, img)
        mask = cv2.resize(mask, (img_size[1], img_size[0]))
        cv2.imwrite("{}/{}.png".format(args.output_dir, img_name), mask)