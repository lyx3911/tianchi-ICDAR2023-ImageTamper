import os
import random
import pickle

#固定随机数种子
random.seed(2023)

def process(dir_path):
    file_name_ls = []
    for file_name in os.listdir(dir_path):
        file_name_ls.append(file_name)
    return file_name_ls

def write_list(list, filename):
    with open(filename, 'w') as f:
        for file in list:
            f.write(file.split('.')[0]+"\n")
    print("save ", filename)


pos_dir = "/data1/datasets/ICDAR2023/train/tampered/imgs/"
neg_dir = "/data1/datasets/ICDAR2023/train/untampered/"
pos_anno_dir = "/data1/datasets/ICDAR2023/train/tampered/masks/"
# neg_anno_dir = "/data1/datasets/ICDAR2023/train/untampered/masks/"
train_img_dir = "/data1/datasets/ICDAR2023/img_dir/train/"
val_img_dir = "/data1/datasets/ICDAR2023/img_dir/val/"
train_mask_dir = "/data1/datasets/ICDAR2023/ann_dir/train/"
val_mask_dir = "/data1/datasets/ICDAR2023/ann_dir/val/"


pos_name_ls = process(pos_dir)
neg_name_ls = process(neg_dir)


random.shuffle(pos_name_ls)
random.shuffle(neg_name_ls)

num_pos = len(pos_name_ls)
num_neg = len(neg_name_ls)


train_pos_name_ls = pos_name_ls[0:int(0.8*num_pos)]
val_pos_name_ls = pos_name_ls[int(0.8*num_pos):]

train_neg_name_ls = neg_name_ls[0:int(0.8*num_neg)]
val_neg_name_ls = neg_name_ls[int(0.8*num_neg):]

write_list(train_pos_name_ls, "splits/train_pos.txt")
write_list(train_neg_name_ls, "splits/train_neg.txt")
write_list(val_pos_name_ls, "splits/val_pos.txt")
write_list(val_neg_name_ls, "splits/val_neg.txt")


new_name = 0
labels = {}


#处理训练集
for name in train_pos_name_ls:
    new_name += 1
    labels[str(new_name)+".jpg"] = 1
    command = "cp " + pos_dir + name +"  " + train_img_dir + str(name)
    os.system(command)
    print(command)

    name = name.split('.')[0]+'.png'
    command = "cp " + pos_anno_dir + name +"  " + train_mask_dir + str(name)
    os.system(command)
    print(command)

# for name in train_neg_name_ls:
#     new_name += 1
#     labels[str(new_name)+".jpg"] = 0
#     command = "cp " + neg_dir + name +"  " + train_dir + str(new_name)+".jpg"
#     os.system(command)
#     print(command)

#处理验证集
for name in val_pos_name_ls:
    new_name += 1
    labels[str(new_name)+".jpg"] = 1
    command = "cp " + pos_dir + name +"  " + val_img_dir + str(new_name)+".jpg"
    os.system(command)
    print(command)

    name = name.split('.')[0]+'.png'
    command = "cp " + pos_anno_dir + name +"  " + val_mask_dir + str(new_name)+".jpg"
    os.system(command)
    print(command)


# for name in val_neg_name_ls:
#     new_name += 1
#     labels[str(new_name)+".jpg"] = 0
#     command = "cp " + neg_dir + name +"  " + val_dir + str(new_name)+".jpg"
#     os.system(command)
#     print(command)

# print(len(labels.keys()))
# with open('datasets/labels.pkl', 'wb') as f:
#     pickle.dump(labels, f)

