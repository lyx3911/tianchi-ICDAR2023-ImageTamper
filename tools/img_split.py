import cv2
from tqdm import tqdm
import os
import math

window_size = 1024
overlap = window_size / 4

def cal_sliding_params(img_h, img_w):
    # 计算需要裁剪成几块
    col, row = 1, 1
    while (window_size*col - (col-1)*overlap) < img_h:
        col += 1
    while (window_size*row - (row-1)*overlap) < img_w:
        row += 1
    return col, row

def img_slide_window(img, mask, col, row):
    imgs = []
    masks = []
    # 计算 overlape
    delta_x, delta_y = 0, 0
    if row > 1:
        delta_x = int((row*window_size-img.shape[1])/(row-1))
    if col > 1:
        delta_y = int((col*window_size-img.shape[0])/(col-1))
    # print(row, col, delta_x, delta_y)    
    
    for i in range(col):
        for j in range(row):
            begin_h = window_size*i - max(0, i)*delta_y
            begin_w = window_size*j - max(0, j)*delta_x
            
            if begin_h + window_size > img.shape[0]:
                begin_h = img.shape[0] - window_size
            if begin_w + window_size > img.shape[1]:
                begin_w = img.shape[1] - window_size
            slide = img[begin_h:begin_h+window_size, begin_w:begin_w+window_size, :]
            mask_slide = mask[begin_h:begin_h+window_size, begin_w:begin_w+window_size, ]
            
            # print(begin_h, begin_w, begin_h+512, begin_w+512, img.shape)
            
            imgs.append(slide)
            masks.append(mask_slide)
            
    return imgs, masks 

if __name__ == '__main__':
    root = "/data1/datasets/ICDAR2023/train/tampered"
    img_files = os.listdir(os.path.join(root, 'imgs'))
    
    save_root = "/data1/datasets/ICDAR2023/train/tampered-train-slide-window-1024/"
    
    for img_file in tqdm(img_files):
        img_name = img_file.split('.')[0]
        # if img_name != '1':
        #     continue
        
        img = cv2.imread(os.path.join(root, "imgs", img_file))
        mask = cv2.imread(os.path.join(root, "masks", img_name+".png"))
        
        h, w, c = img.shape
        if h<window_size or w<window_size:
            size = min(h,w)
            ratio = float(window_size) / float(size)
            if h==w:
                img = cv2.resize(img, (window_size, window_size))
                mask = cv2.resize(mask,(window_size, window_size))
            elif size == h:
                img = cv2.resize(img, (window_size, int(w*ratio)))
                mask = cv2.resize(mask,(window_size, int(w*ratio)))
            else:
                img = cv2.resize(img, (int(h*ratio), window_size))
                mask = cv2.resize(mask, (int(h*ratio), window_size))
            print("resize", img_name, h, w)
        
        
        col, row = cal_sliding_params(img.shape[0], img.shape[1])
        imgs, masks = img_slide_window(img, mask, col, row)
        
        for i, img in enumerate(imgs):
            mask = masks[i]
            # print(img.shape)
            if img.shape[0]!=window_size or img.shape[1]!=window_size:
                print("check:", img_name, h, w)
                exit()
            cv2.imwrite(os.path.join(save_root, "imgs", img_name+"-{}.jpg".format(i)), img)
            cv2.imwrite(os.path.join(save_root, "masks", img_name+"-{}.png".format(i)), mask)                        