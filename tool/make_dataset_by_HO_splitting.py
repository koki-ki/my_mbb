# coding: utf-8
import os
import sys
sys.path.append("../src/lib")
import load_data as ld

OVERWRITE = False
DELIMITER = ","
SEED = 0
BASE_DIR = "../dataset"
RESULT_DIR = "%s/HO" % BASE_DIR

DATASETS = [
    "Abalone01.csv",
    "Abalone.csv",
    "Avila.csv",
    "Banknote.csv",
    "BreastCancer.csv",
    "German.csv",
    "GMM5C.csv",
    "GMM12000.csv",
    "GMM2000.csv",
    "GMM300.csv",
    "Satimage.csv",
    "Spambase.csv",
    "Thyroid.csv",
    "Thyroid2C.csv",
    "WineRed3C.csv",
    "WineWhite3C.csv",
    "Cardiotocography.csv",
    "Cardiotocography2C.csv",
    "Cardio.csv",
    "Ionosphere.csv",
    "Sonar.csv",
]

for DS in DATASETS:
    ds_path = os.path.join(BASE_DIR, DS)
    print (ds_path)

    tr = 0.5
    ts = 1.0 - tr
    if DS == "GMM2000.csv":
        tr = 2000
        ts = 10000
    elif DS == "GMM300.csv":
        tr = 300
        ts = 11700

    ld.load_data(ds_path, delimiter=DELIMITER, training=tr, testing=ts, output_dir=RESULT_DIR,
                 save_split_data=True,
                 load_split_data=True,
                 normalize_by_z_score=True,
                 make_one_hot_label=True,
                 only_training_set=False,
                 overwrite=OVERWRITE,
                 randomize=True,
                 seed=SEED)
