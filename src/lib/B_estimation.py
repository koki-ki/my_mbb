# coding: utf-8
import os
import numpy as np
import tensorflow as tf
from lib.parzen import estimate_scipy_gaussian_kernel_density
import matplotlib.pyplot as plt


def estimate_B_by_weighting(Xtr, ytr, dy):

    # Histogram method
    dy = dy.flatten()
    abs_dy = np.absolute(dy)
    yt = np.argmax(ytr, axis=1)
    anc_y_num_holder = np.zeros(len(np.unique(yt)))
    for i, yc in enumerate(np.unique(yt)):
        abs_dyc = abs_dy[np.where(yt == yc)]
        finite_abs_dyc = abs_dyc[np.where(np.isfinite(abs_dyc))]
        _, bins, _ = plt.hist(finite_abs_dyc, bins="sturges")
        mean_width = np.mean(bins[1:] - bins[:-1])
        anc_y_num = np.count_nonzero((finite_abs_dyc > ((-1.0 * mean_width) / 2.0))
                                     & (finite_abs_dyc < (mean_width / 2.0)))
        anc_y_num_holder[i] = anc_y_num if anc_y_num > 0 else 1

        #plt.rcParams["font.size"] = 60
        #fig = plt.figure(figsize=(32, 16))
        #ax = fig.add_subplot(111)
        #ax_2 = ax.twinx()
        #eps = 10**(-17)
        #ax.hist(abs_dy, color="blue", ec="black", alpha=0.5)
        # ax.set_xlabel("")
        # ax.set_ylabel("")
        ##ax_2.set_xlim((np.amin(abs_dy)-eps, np.amax(abs_dy)+eps))
        #ax_2.set_xlim((np.amin(abs_dy)-eps, 1.5+eps))
        # ax_2.set_ylim(bottom=0.0-0.07)
        # ax_2.set_yticklabels([])
        # ax_2.yaxis.set_ticks_position('none')
        # for ad in sorted(finite_abs_dyc):
        #    ax_2.scatter(ad, -0.02, color="green", s=6000, linewidth=5, edgecolor="black")
        #    if ad > ((-1.0 * mean_width) / 2.0) and ad < (mean_width / 2.0):
        #        ax_2.scatter(ad, -0.02, color="red", s=6000, linewidth=5, edgecolor="black")
        #fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        #fig.savefig(os.path.join("./HIST%s.png" % (i+1)), format="png", dpi=500)

    print(anc_y_num_holder)

    # Extract B
    B = []
    for i, yanc in enumerate(anc_y_num_holder):
        y_ind = np.where(yt == i)[0]
        Xtr_y = Xtr[y_ind]
        NB_score_y = abs_dy[y_ind]
        score_ind = np.argsort(np.array(NB_score_y))[:int(yanc)]
        B_y = Xtr_y[score_ind]
        B.extend(B_y)

    return np.array(B)
