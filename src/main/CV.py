# coding: utf-8

import sys
import os
import argparse
import datetime
import random
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from timeit import default_timer as timer
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
SCRIPT_ABS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_ABS_DIR, "..")))
from lib.load_data import load_data


EXECUTE_DATETIME = datetime.datetime.today().strftime("%Y-%m-%d_%H:%M:%S")


def format_sec_to_hours(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return "%dh:%02dm:%02ds" % (h, m, s)


def save_plt_fig(plt_fig, save_path, clear_fig=True, close_plt_obj=False):
    plt_fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt_fig.savefig(save_path, format="png", dpi=200)
    if clear_fig:
        plt_fig.clf()
    if close_plt_obj:
        plt.close(plt_fig)
    print("Saved plot to %s" % save_path)
    return True


def save_txt(save_path, save_contents, delimiter=",", header=None):
    contents_shape = np.array(save_contents).shape
    if len(contents_shape) != 2:
        print(" * Cannot save to %s (it is not numpy 2D array)" % save_path)
        return False

    conts = []
    if header is not None:
        conts.append(header)
    conts.extend(save_contents)

    with open(save_path, "w") as f:
        for c in conts:
            f.write("%s\n" % delimiter.join(list(map(str, c))))
    print("Saved txt to %s" % save_path)
    return True


def load_txt(load_path, delimiter=","):
    obj = []
    with open(load_path, "r") as f:
        for line in f:
            obj.append(line.strip().split(delimiter))
    print("Loaded txt from %s" % load_path)
    return np.array(obj)


def make_CV_index_list(x, y, cv_fold=160, seed=0):
    np.random.seed(seed)

    y = np.array(y)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    elif len(y.shape > 2):
        print("  *** ERROR: Irregular case of y shape: %s" % y.shape)
        exit(0)

    y_cat, y_cat_N = np.unique(y, return_counts=True)
    y_cat_N_k = y_cat_N // cv_fold
    all_index = list(range(len(y)))
    tr_index_list, ts_index_list, used_ts_index = [], [], []

    for k in range(cv_fold):
        cur_ts_list = []
        for yc in y_cat:
            yc_ind = np.where(y == yc)[0]
            nu_ind = list(set(yc_ind) - set(used_ts_index))
            yc_ts_idx = np.random.choice(
                nu_ind, size=y_cat_N_k[yc], replace=False)
            cur_ts_list.extend(yc_ts_idx)
        used_ts_index.extend(cur_ts_list)

        cur_tr_list = list(set(all_index) - set(cur_ts_list))
        cur_tr_list = np.random.choice(
            cur_tr_list, size=len(cur_tr_list), replace=False)
        tr_index_list.append(cur_tr_list)

        cur_ts_list = np.random.choice(
            cur_ts_list, size=len(cur_ts_list), replace=False)
        ts_index_list.append(cur_ts_list)

        #print("%s - SET TR LENGTH: %s" % (k, len(list(set(np.array(cur_tr_list).flatten())))))
        #print("%s - SET TS LENGTH: %s" % (k, len(list(set(np.array(cur_ts_list).flatten())))))

    tril = np.array(tr_index_list)
    tsil = np.array(ts_index_list)

    #print("TOTAL SET TR LENGTH: %s" % (len(list(set(np.array(cur_tr_list).flatten())))))
    #print("TOTAL SET TS LENGTH: %s" % (len(list(set(np.array(cur_ts_list).flatten())))))
    #print("FINAL SHAPE TR:", tril.shape)
    #print("FINAL SHAPE TS:", tsil.shape)

    return tril, tsil


def CV_SVM(dataset_path, out_dir=None, cv_fold=160, svm_c=range(-15, 16), svm_gamma=range(-15, 16)):

    x, y, _, _ = load_data(dataset_path,
                           delimiter=",",
                           training=1.0,
                           testing=0.0,
                           save_split_data=False,
                           load_split_data=False,
                           normalize_by_z_score=True,
                           make_one_hot_label=True,
                           only_training_set=True,
                           overwrite=False,
                           randomize=False,
                           seed=0)

    dataset_name = os.path.basename(dataset_path).split(".")[0]
    result_list = []

    # Timer Start
    start = timer()

    # C
    for sc in svm_c:
        # gamma
        for sg in svm_gamma:
            # CV Setting
            C = 2**sc
            gamma = 2**sg
            Ltr, Lval = 0.0, 0.0
            train_idx_list, test_idx_list = make_CV_index_list(
                x, y, cv_fold=cv_fold)
            for tril, tsil in zip(train_idx_list, test_idx_list):
                x_train, y_train = x[tril], y[tril]
                x_test, y_test = x[tsil], y[tsil]
                y_train, y_test = np.argmax(
                    y_train, axis=1), np.argmax(y_test, axis=1)

                # SVM Training
                cls = SVC(C=C, gamma=gamma)
                cls.fit(x_train, y_train)

                y_pred_train, y_pred_test = cls.predict(
                    x_train), cls.predict(x_test)

                Ltr += 1.0 - accuracy_score(y_train, y_pred_train)
                Lval += 1.0 - accuracy_score(y_test, y_pred_test)

            Ltr /= cv_fold
            Lval /= cv_fold

            result_list.append([sc, sg, Ltr, Lval])

            print("CV_SVM %s - Etr:%.5f, Eval:%.5f (C:%s, gamma:%s)" %
                  (dataset_name, Ltr, Lval, C, gamma))

    # Timer Stop
    end = timer()

    result_list = np.array(result_list)

    if out_dir is None:
        out_dir = os.path.dirname(dataset_path)
    image_dir = os.path.join(out_dir, "image")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)
    res_basename = "CV_SVM_%s" % dataset_name

    # Important info.
    min_Tr_index = np.argmin(result_list[:, 2])
    min_Ts_index = np.argmin(result_list[:, 3])
    min_Tr, min_Tr_C, min_Tr_gamma = result_list[min_Tr_index,
                                                 2], result_list[min_Tr_index, 0], result_list[min_Tr_index, 1]
    min_Ts, min_Ts_C, min_Ts_gamma = result_list[min_Ts_index,
                                                 3], result_list[min_Ts_index, 0], result_list[min_Ts_index, 1]

    print("CV_SVM %s: Minimum Etr: %.5f (C:2**%s, gamma:2**%s), Minimum Eval: %.5f (C:2**%s, gamma:2**%s)" %
          (dataset_name, min_Tr, min_Tr_C, min_Tr_gamma, min_Ts, min_Ts_C, min_Ts_gamma))

    info = []
    info.append(["Dataset", dataset_name])
    info.append(["Date", EXECUTE_DATETIME])
    info.append(["CV: fold k", cv_fold])
    info.append(["SVM: C range (2**)", str(svm_c)])
    info.append(["SVM: gamma range (2**)", str(svm_gamma)])
    info.append(["CV_SVM: Minimum training error (C=%s gamma=%s)" %
                (min_Tr_C, min_Tr_gamma), min_Tr])
    info.append(["CV_SVM: Minimum validation error (C=%s gamma=%s)" %
                (min_Ts_C, min_Ts_gamma), min_Ts])
    info.append(["Runtime", format_sec_to_hours(float(end - start))])
    save_txt(os.path.join(out_dir, "log_%s.txt" % res_basename), info)

    # Plotting info.
    seq_param_C = result_list[:, 0]
    seq_param_gamma = result_list[:, 1]
    seq_Etr = result_list[:, 2].reshape(len(svm_c), len(svm_gamma))
    seq_Ets = result_list[:, 3].reshape(len(svm_c), len(svm_gamma))

    extent = [seq_param_gamma[0], seq_param_gamma[-1],
              seq_param_C[-1], seq_param_C[0]]

    plt.rcParams["font.size"] = 32
    fig = plt.figure(figsize=(32, 24))
    ax = fig.add_subplot(111)
    ax.set_xlabel("gamma (2**)")
    ax.set_xticks(seq_param_gamma)
    ax.set_ylabel("C (2**)")
    ax.set_yticks(seq_param_C)
    im = ax.imshow(seq_Etr, cmap="cool", extent=extent, aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.scatter(min_Tr_gamma, min_Tr_C, color="r", marker="s", s=1000)
    fig.suptitle("CV_SVN %s: Minimum Etr: %.5f (C:2**%s, gamma:2**%s)" %
                 (dataset_name, min_Tr, min_Tr_C, min_Tr_gamma))
    save_plt_fig(fig, os.path.join(image_dir, "%s_Etr.png" % res_basename))

    ax = fig.add_subplot(111)
    ax.set_xlabel("gamma (2**)")
    ax.set_xticks(seq_param_gamma)
    ax.set_ylabel("C (2**)")
    ax.set_yticks(seq_param_C)
    im = ax.imshow(seq_Ets, cmap="cool", extent=extent, aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.scatter(min_Ts_gamma, min_Ts_C, color="r", marker="s", s=1000)
    fig.suptitle("CV_SVN %s: Minimum Eval: %.5f (C:2**%s, gamma:2**%s)" %
                 (dataset_name, min_Ts, min_Ts_C, min_Ts_gamma))
    save_plt_fig(fig, os.path.join(image_dir, "%s_Eval.png" % res_basename))

    # All result
    save_txt(os.path.join(out_dir, "%s.csv" % res_basename), result_list,
             header=["SVM_C(2**)", "SVM_gamma(2**)", "Etr", "Eval"])


def CV_PBC(dataset_path, out_dir=None, cv_fold=160, pbc_p=range(1, 101)):

    x, y, _, _ = load_data(dataset_path,
                           delimiter=",",
                           training=1.0,
                           testing=0.0,
                           save_split_data=False,
                           load_split_data=False,
                           normalize_by_z_score=True,
                           make_one_hot_label=True,
                           only_training_set=True,
                           overwrite=False,
                           randomize=False,
                           seed=0)

    dataset_name = os.path.basename(dataset_path).split(".")[0]
    result_list = []

    y_c, N_y_c = np.unique(np.argmax(y, axis=1), return_counts=True)
    if min(N_y_c) < max(pbc_p):
        print("   *** WARNINIG: Max prototype num should be set %s (The minimum sample num of each class)" % min(N_y_c))
        pbc_p = range(1, min(N_y_c))

    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
    )

    # Timer Start
    start = timer()

    # p
    for p in pbc_p:
        Ltr, Lval = 0.0, 0.0
        train_idx_list, test_idx_list = make_CV_index_list(
            x, y, cv_fold=cv_fold)
        for tril, tsil in zip(train_idx_list, test_idx_list):
            x_train, y_train = x[tril], y[tril]
            x_test, y_test = x[tsil], y[tsil]

            class_num = y_train.shape[1]
            dim_num = x_train.shape[1]
            y_target = tf.placeholder(
                shape=[None, class_num], dtype=tf.float32)
            x_data = tf.placeholder(shape=[None, dim_num], dtype=tf.float32)
            kmeans = KMeans(n_clusters=p, n_init=20, random_state=0)
            proto = np.array([kmeans.fit(x_train[np.where(np.argmax(
                y_train, axis=1) == cls)]).cluster_centers_ for cls in range(class_num)])
            A = tf.Variable(proto, dtype=tf.float32)
            g = tf.reduce_sum(tf.map_fn(lambda x: tf.subtract(
                tf.multiply(A, x), tf.multiply(0.5, tf.square(A))), x_data), axis=3)
            nearest_proto = tf.reduce_max(g, axis=2)
            y_pred = tf.argmax(nearest_proto, axis=1)  # n
            y_true = tf.argmax(y_target, axis=1)  # n
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(y_pred, y_true), tf.float32))  # 1
            error = tf.subtract(1.0, accuracy)

            with tf.Session(graph=tf.get_default_graph(), config=session_conf) as sess:
                sess.run(tf.global_variables_initializer())
                closed_error = sess.run(
                    error, feed_dict={x_data: x_train, y_target: y_train})
                open_error = sess.run(
                    error, feed_dict={x_data: x_test, y_target: y_test})

            Ltr += closed_error
            Lval += open_error

        Ltr /= cv_fold
        Lval /= cv_fold

        print("CV_PBC %s - Etr:%.5f, Eval:%.5f (p:%s)" %
              (dataset_name, Ltr, Lval, p))

        result_list.append([p, Ltr, Lval])

    # Timer Stop
    end = timer()

    result_list = np.array(result_list)

    if out_dir is None:
        out_dir = os.path.dirname(dataset_path)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    image_dir = os.path.join(out_dir, "image")
    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)
    res_basename = "CV_PBC_%s" % dataset_name

    # Important info.
    min_Tr_index = np.argmin(result_list[:, 1])
    min_Ts_index = np.argmin(result_list[:, 2])
    min_Tr_p, min_Tr = result_list[min_Tr_index,
                                   0], result_list[min_Tr_index, 1]
    min_Ts_p, min_Ts = result_list[min_Ts_index,
                                   0], result_list[min_Ts_index, 2]

    print("CV_PBC %s: Minimum Etr: %.5f (p:%s), Minimum Eval: %.5f (p:%s)" %
          (dataset_name, min_Tr, min_Tr_p, min_Ts, min_Ts_p))

    info = []
    info.append(["Dataset", dataset_name])
    info.append(["Date", EXECUTE_DATETIME])
    info.append(["CV: fold k", cv_fold])
    info.append(["PBC: p range", str(pbc_p)])
    info.append(["CV_PBC: Minimum training error (p=%s)" % (min_Tr_p), min_Tr])
    info.append(["CV_PBC: Minimum validation error (p=%s)" %
                (min_Ts_p), min_Ts])
    info.append(["Runtime", format_sec_to_hours(float(end - start))])
    save_txt(os.path.join(out_dir, "log_%s.txt" % res_basename), info)

    plt.rcParams["font.size"] = 32
    fig = plt.figure(figsize=(32, 24))
    ax = fig.add_subplot(111)
    ax.set_xlabel("# p")
    ax.set_xticks(pbc_p)
    ax.set_ylabel("Error Rate")
    ax.plot(result_list[:, 0], result_list[:, 1],
            c="red", linewidth=15, label="Training Error")
    ax.plot(result_list[:, 0], result_list[:, 2],
            c="orange", linewidth=15, label="Validating Error")
    ax.legend(loc="best")
    fig.suptitle("CV_PBC %s: Min Etr: %.5f (p:%s), Min Eval: %.5f (p:%s)" % (
        dataset_name, min_Tr, min_Tr_p, min_Ts, min_Ts_p))
    save_plt_fig(fig, os.path.join(image_dir, "%s_EtrEval.png" % res_basename))

    # All result
    save_txt(os.path.join(out_dir, "%s.csv" % res_basename),
             result_list, header=["PBC_p", "Etr", "Eval"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog=SCRIPT_NAME)
    parser.add_argument("-d", "--dataset", help="The path of dataset (default: ../../dataset/GMM5C.csv)",
                        type=str, default="../../dataset/GMM5C.csv")
    parser.add_argument(
        "-o", "--output", help="The path to output directory (default: ../../res/CV", type=str, default="../../res/CV")
    parser.add_argument("--fold", help="", type=int, default=10)
    parser.add_argument("--min-p", help="", type=str, default="p1")
    parser.add_argument("--max-p", help="", type=str, default="p100")
    parser.add_argument("--min-s", help="", type=str, default="p15")
    parser.add_argument("--max-s", help="", type=str, default="m15")
    parser.add_argument("-m", "--mode", help="", type=str, default="ALL")

    args = parser.parse_args()
    dataset_path = args.dataset
    output_directory = args.output
    fold = args.fold
    min_pbc_num = int(
        args.min_p[1:]) if args.min_p[0] == "p" else -int(args.min_p[1:])
    max_pbc_num = int(
        args.max_p[1:]) if args.max_p[0] == "p" else -int(args.max_p[1:])
    min_svm_num = int(
        args.min_s[1:]) if args.min_s[0] == "p" else -int(args.min_s[1:])
    max_svm_num = int(
        args.max_s[1:]) if args.max_s[0] == "p" else -int(args.max_s[1:])
    mode = args.mode

    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)

    if mode == "PBC" or mode == "ALL":
        CV_PBC(dataset_path, out_dir=output_directory, cv_fold=fold,
               pbc_p=range(min_pbc_num, max_pbc_num+1))
    if mode == "SVM" or mode == "ALL":
        CV_SVM(dataset_path, out_dir=output_directory, cv_fold=fold, svm_c=range(
            min_svm_num, max_svm_num+1), svm_gamma=range(min_svm_num, max_svm_num+1))
