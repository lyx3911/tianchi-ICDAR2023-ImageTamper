import numpy as np
import os
import cv2
from tqdm import tqdm
import os
import random
import pickle
from collections import Counter

#固定随机数种子
random.seed(2023)
#with open('dataset/datasets_pkl', 'rb') as f:
#    datasets_pkl = pickle.load(f)

#注意下这里填入测试集文件夹地址
test_dir = "imgs/"

test = []
for file_name in os.listdir(test_dir):
    file_path = test_dir + "/" + file_name
    if "jpg" not in file_path:
        print(file_path)
        continue
    test.append((file_path, file_name))


datasets_pkl = {}
datasets_pkl["test"] = test

with open('dataset/datasets_pkl', 'wb') as f:
    pickle.dump(datasets_pkl, f)
