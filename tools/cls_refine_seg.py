import pandas as pd
import os
import cv2
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    labels = pd.read_csv('./csv/result-81.92.csv', delim_whitespace=True, header=None).to_numpy()
    
    tampers = labels[labels[:,1]>=0.5] 
    untampers = labels[labels[:,1]<0.5]
    # print(tampers.shape, untampers.shape)
    
    root = "output/convnext512-train1024+768slide+all-infer512-alldata/result/"

    files = os.listdir(root)
    # files = [file.split('.')[0] for file in files]
    
    for f in tqdm(untampers):
        name = f[0].split('.')[0]
        mask = cv2.imread(root+name+'.png', cv2.IMREAD_GRAYSCALE)
        # print(root+name+'.png')
        # mask = np.clip(mask, 0, int(255*f[1]))
        mask = np.uint8(mask * f[1])
        cv2.imwrite("output/convnext512-train1024+768slide+all-infer512-alldata/submission/{}.png".format(name), mask)
        # print("../output/unetplusplus-swinv2-512/cls_refine_untampered_clip/submission/{}.png".format(name))
    for f in tqdm(tampers):
        name = f[0].split('.')[0]
        mask = cv2.imread(root+name+'.png', cv2.IMREAD_GRAYSCALE)
        # print(root+name+'.png')
        cv2.imwrite("output/convnext512-train1024+768slide+all-infer512-alldata/submission/{}.png".format(name), mask)