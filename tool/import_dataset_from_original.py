# coding: utf-8
import os
import sys

DATASETS = {}
DATASETS["abalone_01.dat"] = "Abalone01.csv"
DATASETS["abalone.dat"] = "Abalone.csv"
DATASETS["avila.dat"] = "Avila.csv"
DATASETS["banknote.dat"] = "Banknote.csv"
DATASETS["breast_cancer.dat"] = "BreastCancer.csv"
DATASETS["german.dat"] = "German.csv"
DATASETS["GMM_5c_1000.dat"] = "GMM5C.csv"
DATASETS["GMM_tr_te.dat"] = "GMM12000.csv"
DATASETS["GMM_tr_te_2000.dat"] = "GMM2000.csv"
DATASETS["GMM_tr_te_300.dat"] = "GMM300.csv"
DATASETS["satimage.dat"] = "Satimage.csv"
DATASETS["spambase.dat"] = "Spambase.csv"
DATASETS["thyroid.dat"] = "Thyroid.csv"
DATASETS["thyroid_23.dat"] = "Thyroid2C.csv"
DATASETS["wine_red_merged.dat"] = "WineRed3C.csv"
DATASETS["wine_white_merged.dat"] = "WineWhite3C.csv"
DATASETS["cardiotocography.dat"] = "Cardiotocography.csv"
DATASETS["cardiotocography_12.dat"] = "Cardiotocography2C.csv"
DATASETS["cardio.dat"] = "Cardio.csv"
DATASETS["ionosphere.dat"] = "Ionosphere.csv"
DATASETS["sonar.dat"] = "Sonar.csv"

FROM_DIR = "../../ETP/dataset/data/"
TO_DIR = "."
for bd in os.listdir(FROM_DIR):
    if not bd in DATASETS.keys():
        continue
    from_path = os.path.join(FROM_DIR, bd)
    print(from_path)
    dat = []
    with open(from_path, "r") as f:
        for line in f:
            conts = line.strip().split(" ")
            dat.append(conts)
    to_path = os.path.join(TO_DIR, DATASETS[bd])
    print(to_path)
    with open(to_path, "w") as f:
        for d in dat:
            f.write("%s\n" % ",".join(d))
