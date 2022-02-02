# coding: utf-8
import os
import datetime
import numpy as np
from matplotlib import pyplot as plt

SCRIPT_NAME = os.path.basename(os.path.splitext(__file__)[0])
EXECUTE_DATETIME = datetime.datetime.today().strftime("%Y-%m-%d_%H:%M:%S")
RES_DIR = "../res"
LOG_DIR = os.path.join(RES_DIR, SCRIPT_NAME)
IMAGE_DIR = os.path.join(LOG_DIR, "image")
if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

def save_txt(save_path, save_contents, delimiter=",", header=None):
    contents_shape = np.array(save_contents).shape
    if len(contents_shape) != 2:
        print (" * Cannot save to %s (it is not numpy 2D array)" % save_path)
        return False

    conts = []
    if header is not None:
        conts.append(header)
    conts.extend(save_contents)

    with open(save_path, "w") as f:
        for c in conts:
            f.write("%s\n" % delimiter.join(list(map(str, c))))
    print ("Saved txt to %s" % save_path)
    return True

def save_plt_fig(plt_fig, save_path, clear_fig=True, close_plt_obj=False):
    plt_fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt_fig.savefig(save_path, format="png", dpi=200)
    if clear_fig:
        plt_fig.clf()
    if close_plt_obj:
        plt.close(plt_fig)
    print ("Saved plot to %s" % save_path)
    return True

def generate_2D_samples(data_num=1000, m1=1, m2=1, s11=1, s12=1, s21=1, s22=1, class_label=1):
    D = []
    for i in range(data_num):
        mu = [m1, m2]
        cov = [[s11, s12],
               [s21, s22]]
        X = np.random.multivariate_normal(mu, cov, 1)[0].tolist()
        X.append(class_label)
        D.append(X)
    return D

if __name__ == "__main__":

    seed = 0
    np.random.seed(seed=seed)

    # Setting config
    DATASET_NAME = "GMM6C"
    GMM_PARAMS = [
        [1000, -3, -3, 2, 0, 0, 2], # [N, m1, m2, s11, s12, s21, s22]
        [1000, 0, 0, 2, 0, 0, 2], # [N, m1, m2, s11, s12, s21, s22]
        [1000, 2, 2, 2, 0, 0, 2], # [N, m1, m2, s11, s12, s21, s22]
        [1000, -2, 1, 2, 0, 0, 2], # [N, m1, m2, s11, s12, s21, s22]
        [1000, 3, -1, 2, 0, 0, 2], # [N, m1, m2, s11, s12, s21, s22]
        [1000, 0, -4, 2, 0, 0, 2], # [N, m1, m2, s11, s12, s21, s22]
    ]

    # Generating
    D = []
    for i, GP in enumerate(GMM_PARAMS):
        d = generate_2D_samples(GP[0], GP[1], GP[2], GP[3], GP[4], GP[5], GP[6], i)
        D.extend(d)
    D = np.array(D)
    save_txt(os.path.join(LOG_DIR, "%s.csv" % DATASET_NAME), D, delimiter=",")

    # Plotting
    y = D[:, -1]
    yc, yc_N = np.unique(y, return_counts=True)

    plt.rcParams["font.size"] = 40
    fig = plt.figure(figsize=(32, 16), edgecolor="black", linewidth=5)
    ax = fig.add_subplot(111)
    ax.set_title("%s: D=%s, J=%s N=%s %s"
                 % (DATASET_NAME, D.shape[1]-1, len(yc), D.shape[0], str(yc_N)))
    cmap = list(map(plt.get_cmap("tab10"), D[:, 2].astype(int)))
    ax.scatter(D[:, 0], D[:, 1], marker="o", s=100, c=cmap)
    save_plt_fig(fig, os.path.join(IMAGE_DIR, "%s.png" % DATASET_NAME),
                 clear_fig=True, close_plt_obj=False)

    # Information
    info = []
    info.append(["Dataset", DATASET_NAME])
    info.append(["Create date", EXECUTE_DATETIME])
    info.append(["Save path", os.path.join(LOG_DIR, "%s.csv" % DATASET_NAME)])
    info.append(["Seed", seed])
    info.append(["N", D.shape[0]])
    info.append(["D", D.shape[1]-1])
    info.append(["J", len(yc)])
    for i, c in enumerate(yc):
        info.append(["N_%s" % c, yc_N[i]])
    save_txt(os.path.join(LOG_DIR, "log_%s.txt" % DATASET_NAME), info, delimiter=",")

