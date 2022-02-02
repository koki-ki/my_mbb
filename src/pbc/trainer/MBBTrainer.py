# config: utf-8
import os
import numpy as np
import pandas as pd
import sys

SCRIPT_NAME = os.path.splitext(__file__)[0]
SCRIPT_ABS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_ABS_DIR, "../../")))
sys.path.append('../../')

from pbc.base.base import BaseTrain
from lib.B_estimation import estimate_B_by_weighting
from lib.P_estimation import estimate_P_on_B_segmentation
from lib.parzen import estimate_scipy_gaussian_kernel_density, estimate_one_class_parzen_width

import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 24
CMAP = plt.get_cmap("tab10")

def save_plt_fig(plt_fig, save_path, clear_fig=True, close_plt_obj=False):
    # plt_fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt_fig.savefig(save_path, format="png", dpi=200)
    if clear_fig:
        plt_fig.clf()
    if close_plt_obj:
        plt.close(plt_fig)
    print ("Saved plot to %s" % save_path)
    return True


class MBBTrainer(BaseTrain):
    def __init__(self, classifier, trainer, optimizer, sess, conf, Xtr, ytr, Xts, yts):
        super(MBBTrainer, self).__init__(classifier, trainer, optimizer, sess, conf)

        self.Xtr = Xtr
        self.ytr = ytr
        self.Xts = Xts
        self.yts = yts

        self.yt = self.sess.run(self.classifier.y_true, feed_dict={self.classifier.y: self.ytr})
        self.feed_dict_train = {self.classifier.x: self.Xtr, self.classifier.y: self.ytr}
        self.feed_dict_test = {self.classifier.x: self.Xts, self.classifier.y: self.yts}


    def show_result_series(self):
        df_res = pd.DataFrame.from_dict(self.result)
        n_epoch = df_res["Epoch"]
        n_loss = df_res["Loss"]
        min_loss = n_loss.min()
        fin_loss = n_loss.iloc[-1]
        n_Etr = df_res["TrainingError"]
        min_Etr = n_Etr.min()
        fin_Etr = n_Etr.iloc[-1]
        n_Ets = df_res["TestingError"]
        min_Ets = n_Ets.min()
        fin_Ets = n_Ets.iloc[-1]
        n_PhatBar_j = [df_res["PhatBar_%s" % (j+1)] for j in range(self.classifier.class_num)]
        n_lenS = df_res["S"]
        n_lenS_j = [df_res["S_%s" % (j+1)] for j in range(self.classifier.class_num)]

        fig3216 = plt.figure(figsize=(32, 16))
        ax221 = fig3216.add_subplot(221)
        ax221.plot(n_epoch, n_loss, color="green", lw=3, label="Loss: Min.=%.9f, Last=%.9f" % (min_loss, fin_loss))
        ax221.legend(loc="best")

        ax222 = fig3216.add_subplot(222)
        for j in range(self.classifier.class_num):
            ax222.plot(n_epoch, n_PhatBar_j[j], color=CMAP(j), lw=3, label="PhatBar_%s" % (j+1))
        ax222.legend(loc="best")

        ax223 = fig3216.add_subplot(223)
        ax223.plot(n_epoch, n_Etr, color="red", lw=3, label="TrainingError: Min.=%.9f, Last=%.9f" % (min_Etr, fin_Etr))
        ax223.plot(n_epoch, n_Ets, color="orange", lw=3, label="TestingError: Min.=%.9f, Last=%.9f" % (min_Ets, fin_Ets))
        ax223.legend(loc="best")

        ax224 = fig3216.add_subplot(224)
        ax224.plot(n_epoch, n_lenS, color="purple", lw=3, label="S")
        for j in range(self.classifier.class_num):
            ax224.plot(n_epoch, n_lenS_j[j], color=CMAP(j), lw=3, label="S_%s" % (j+1))
        ax224.legend(loc="best")

        save_plt_fig(fig3216, os.path.join(self.conf.output, "SERIES.png"))
        plt.close(fig3216)


    def train_epoch(self, current_epoch):
        self.yp = self.sess.run(self.classifier.y_pred, feed_dict=self.feed_dict_train)

        if (current_epoch % self.conf.reest_B_interval) == 0 or current_epoch == self.conf.epoch:
            self.update_B(current_epoch)

        if (current_epoch % self.conf.reest_P_interval) == 0 or current_epoch == self.conf.epoch:
            self.update_P(current_epoch)

        # if (self.classifier.dim_num == 2) and (current_epoch % self.conf.plotting_interval) == 0:
            # print ("2D Plotting ...")

            # A = self.sess.run(self.classifier.A)
            # ytS = self.yt[self.flatten_S]
            # xS = self.Xtr[self.flatten_S]
            # wS = self.sess.run(self.trainer.W, feed_dict=self.feed_dict_training_S).flatten()
            # PhatS = self.sess.run(self.trainer.P_hat, feed_dict=self.feed_dict_training_S).flatten()
            # cntS = self.sess.run(self.trainer.S_cnt, feed_dict=self.feed_dict_training_S).flatten()

            # A_by_class = [A[i, :, :] for i in range(self.classifier.class_num)]
            # Xtr_yp_by_class = [self.Xtr[np.where(self.yp == i)] for i in range(self.classifier.class_num)]
            # Xtr_yt_by_class = [self.Xtr[np.where(self.yt == i)] for i in range(self.classifier.class_num)]
            # xS_yt_by_class = [xS[np.where(ytS == i)] for i in range(self.classifier.class_num)]
            # wS_by_class = [wS[np.where(ytS == i)] for i in range(self.classifier.class_num)]
            # PhatS_by_class = [PhatS[np.where(ytS == i)] for i in range(self.classifier.class_num)]

            # fig3216 = plt.figure(figsize=(32, 16))

            # # Sample label
            # ax = fig3216.add_subplot(1, 2, 1)
            # for j in range(self.classifier.class_num):
            #     ax.scatter(Xtr_yt_by_class[j][:, 0], Xtr_yt_by_class[j][:, 1],
            #                marker="o", s=120, c=CMAP(j), alpha=0.5, label="class %s" % (j+1))
            # for j in range(self.classifier.class_num):
            #     ax.scatter(A_by_class[j][:, 0], A_by_class[j][:, 1],
            #                marker="s", s=180, c=CMAP(j), linewidth=3, edgecolors="black", label="class %s" % (j+1))

            # ax = fig3216.add_subplot(1, 2, 2)
            # for j in range(self.classifier.class_num):
            #     ax.scatter(Xtr_yp_by_class[j][:, 0], Xtr_yp_by_class[j][:, 1],
            #                marker="o", s=120, c=CMAP(j), alpha=0.5, label="class %s" % (j+1))
            # for j in range(self.classifier.class_num):
            #     ax.scatter(A_by_class[j][:, 0], A_by_class[j][:, 1],
            #                marker="s", s=180, c=CMAP(j), linewidth=3, edgecolors="black", label="class %s" % (j+1))

            # save_plt_fig(fig3216, os.path.join(self.conf.output, "Prototypes-%s.png" % current_epoch))

            # # Phat and Weight
            # ax = fig3216.add_subplot(1, 2, 1)
            # for j in range(self.classifier.class_num):
            #     ax.scatter(Xtr_yp_by_class[j][:, 0], Xtr_yp_by_class[j][:, 1],
            #                marker="o", s=120, c=CMAP(j), alpha=0.5, label="class %s" % (j+1))
            # for j in range(self.classifier.class_num):
            #     axc1 = ax.scatter(xS_yt_by_class[j][:, 0], xS_yt_by_class[j][:, 1],
            #                       marker="o", s=180, linewidth=3, edgecolors="black", c=PhatS_by_class[j], cmap="magma", vmin=0.0, vmax=1.0)
            # fig3216.colorbar(axc1)

            # ax = fig3216.add_subplot(1, 2, 2)
            # for j in range(self.classifier.class_num):
            #     ax.scatter(Xtr_yp_by_class[j][:, 0], Xtr_yp_by_class[j][:, 1],
            #                marker="o", s=120, c=CMAP(j), alpha=0.5, label="class %s" % (j+1))
            # for j in range(self.classifier.class_num):
            #     axc2 = ax.scatter(xS_yt_by_class[j][:, 0], xS_yt_by_class[j][:, 1],
            #                       marker="o", s=180, linewidth=3, edgecolors="black", c=wS_by_class[j], cmap="magma")
            # fig3216.colorbar(axc2)

            # save_plt_fig(fig3216, os.path.join(self.conf.output, "PhatWeights-%s.png" % current_epoch))
            # plt.close(fig3216)

        # TRAINING
        self.train_step(current_epoch)
        #
        current_result = {"Epoch": current_epoch,
                          "Loss": self.loss,
                          "TrainingError": self.tr_err,
                          "TestingError": self.ts_err}

        Phat = self.sess.run(self.trainer.P_hat, feed_dict=self.feed_dict_training_S).flatten()
        PhatBar_by_class = [np.mean(Phat[np.where(self.yt[self.flatten_S] == j)], axis=0) for j in range(self.classifier.class_num)]
        for j in range(self.classifier.class_num):
            current_result["PhatBar_%s" % (j+1)] = PhatBar_by_class[j]
        current_result["S"] = len(self.flatten_S)
        len_S_by_class = [len(np.where(self.yt[self.flatten_S] == j)[0]) for j in range(self.classifier.class_num)]
        for j in range(self.classifier.class_num):
            current_result["S_%s" % (j+1)] = len_S_by_class[j]

        # print ("%s: %s" % (self.conf.name, str(current_result)))

        """
        aaa = self.sess.run(self.trainer.a, feed_dict=self.feed_dict_training_S).flatten()
        lll = self.sess.run(self.trainer.l, feed_dict=self.feed_dict_training_S).flatten()
        HHH = self.sess.run(self.trainer.H_hat, feed_dict=self.feed_dict_training_S).flatten()
        hwS = self.hw_S.flatten()
        dyS = self.sess.run(self.trainer.dy_S, feed_dict=self.feed_dict_training_S).flatten()
        for hws, dys, P, a, H, l in sorted(zip(hwS, dyS, Phat, aaa, HHH, lll), key=lambda x: x[3]):
            print ("%s - %s - %s - %s - %s - %s" % (hws, dys, P, a, H, l))
        #"""

        self.result.append(current_result)


    def train_step(self, current_epoch):
        # Training
        self.feed_dict_epoch = {self.optimizer.epsilon: self.conf.learning_rate,
                                self.optimizer.epoch: current_epoch % self.conf.reest_B_interval}
        _ = self.sess.run(self.optimizer.A_next,
                          feed_dict={**self.feed_dict_training_S, **self.feed_dict_epoch})

        # Evaluating
        self.loss = self.sess.run(self.trainer.L, feed_dict=self.feed_dict_training_S)
        self.tr_err = self.sess.run(self.classifier.error, feed_dict=self.feed_dict_train)
        self.ts_err = self.sess.run(self.classifier.error, feed_dict=self.feed_dict_test)

        """
        aaa = self.sess.run(self.trainer.aaa, feed_dict=self.feed_dict_training_S)
        print(aaa)
        print(aaa.shape)
        exit()
        """


    def update_B(self, current_epoch):
        self.dy = self.sess.run(self.classifier.dy, feed_dict=self.feed_dict_train)
        self.B = estimate_B_by_weighting(self.Xtr, self.ytr, self.dy)

    def update_P(self, current_epoch):
        self.S, self.g_seg_ind, self.seg_map_h, self.seg_map_y = \
            estimate_P_on_B_segmentation(self.sess,
                                         self.classifier,
                                         self.B,
                                         self.Xtr,
                                         self.ytr,
                                         self.dy,
                                         seg_size=self.conf.knn_size)

        self.flatten_S = self.S.flatten()
        self.hw_S = estimate_scipy_gaussian_kernel_density(self.dy[self.flatten_S])  # scipy-parzen
        #self.hw_S = estimate_one_class_parzen_width(self.dy[self.flatten_S])  # loo-parzen
        self.feed_dict_training_S = {self.classifier.x: self.Xtr,
                                     self.classifier.y: self.ytr,
                                     self.trainer.h_seg_map: self.seg_map_h,
                                     self.trainer.y_seg_map: self.seg_map_y,
                                     self.trainer.g_seg_ind: self.g_seg_ind,
                                     self.trainer.S_ind: self.S,
                                     self.trainer.h_S: self.hw_S}

