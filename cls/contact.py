import pandas as pd

import pickle
import numpy as np



result1 = pd.read_csv("512.csv",header=None,sep="\t").to_numpy()
result2 = pd.read_csv("768.csv",header=None,sep="\t").to_numpy()

preds = []
ids = []

for val1, val2 in zip(result1, result2):
    id1, score1 = val1
    id2, score2 = val2

    if id1 != id2:
        print("error")
        exit(0)
    
    ids.append(id1)
    preds.append((score1*0.5 + score2*0.5))


with open("result.csv","w",encoding="utf-8") as f:
    for ID, pred in zip(ids, preds):
        f.writelines(str(ID)+"\t"+str(pred)+"\n")


