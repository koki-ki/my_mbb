# config: utf-8
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 40

def get_df_index(df, idx=-1):
    if isinstance(idx, int):
        return idx
    if isinstance(idx, str):
        if idx == "Min":
            return df.idxmin()
        elif idx == "Max":
            return df.idxmax()
        else:
            return -1

def read_by_pandas(file_path):
    return pd.read_csv(file_path)

def read_CV_result(file_path, delimiter=","):
    if not os.path.isfile(file_path):
        return []
    res = []
    with open(file_path, "r") as f:
        header = f.readline()
        for line in f:
            conts = line.strip().split(delimiter)
            conts = list(map(float, conts))
            res.append(conts)
    return np.array(res)

def save_plt_fig(plt_fig, save_path, clear_fig=True, close_plt_obj=False):
    plt_fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt_fig.savefig(save_path, format="png", dpi=200, edgecolor="black")
    if clear_fig:
        plt_fig.clf()
    if close_plt_obj:
        plt.close(plt_fig)
    print ("Saved plot to %s" % save_path)
    return True

SCRIPT_NAME = os.path.basename(os.path.splitext(__file__)[0])
RES_DIR = "../res/%s" % SCRIPT_NAME
if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)

target_lr_list = ["0p01", "0p05", "0p1", "0p5", "1p0"]
target_ver_list = ["master", "DEBUGGING"]
target_ver_list.append("on-boundary-d")
#target_ver_list.append("on-boundary-d-proto120")

target_col = "Loss"  # header: loss=0, ..
#target_col = "TestingError"  # header: loss=0, ..
#target_row = "Min"  # int()=index, str()='Min' or 'Max'
target_row = -1  # int()=index, str()='Min' or 'Max'

#show_numerials = True
show_numerials = False

LR_DIR = "../res/MBBTraining"
CV_DIR = "../res/exp_CV"

# Read training result
tr_res = []
for tr_ver in os.listdir(LR_DIR):
    tr_ver_path = os.path.join(LR_DIR, tr_ver)  # (git version name)
    if os.path.isfile(tr_ver_path):
        continue

    if not tr_ver in target_ver_list:
        continue

    for ds in os.listdir(tr_ver_path):
        ds_path = os.path.join(tr_ver_path, ds)  # (dataset name)
        if os.path.isfile(ds_path):
            continue

        target_df = None  # target dataframe (pandas) <- use later
        target_val = None  # target value (minimum)
        target_lr = None  # target learning rate value (to get purpose LR)
        target_row_idx = None
        target_col_idx = None

        for lr in sorted(os.listdir(ds_path)):  # (learning rate)
            lr_path = os.path.join(ds_path, lr)
            res_path = os.path.join(lr_path, "result.csv")
            if not os.path.isfile(res_path):
                continue

            LR_str = str(lr.split("_")[1].strip("LR"))
            if not LR_str in target_lr_list:
                continue

            cur_res_df = read_by_pandas(res_path)
            cur_col_idx = cur_res_df.columns.get_loc(target_col)
            cur_res_series = cur_res_df.iloc[:, cur_col_idx]
            cur_row_idx = get_df_index(cur_res_series, target_row)
            cur_res_val = cur_res_series.iloc[cur_row_idx]

            if target_df is None or cur_res_val <= target_val:
                target_df = cur_res_df
                target_val = cur_res_val
                target_lr = float(LR_str.replace("p", "."))
                target_row_idx = cur_row_idx
                target_col_idx = cur_col_idx

        if not target_df is None:
            # [training_version, dataset_name, learning_rate, dataframe, row_idx, col_idx]
            tr_res.append([tr_ver, ds, target_lr, target_df, target_row_idx, target_col_idx])

# Read CV result
CV_PBC_res, CV_SVM_res = [], []
for tr in tr_res:
    ds_name = tr[1]
    CV_PBC_path = os.path.join(CV_DIR, "CV_PBC_%s.csv" % ds_name)
    if os.path.isfile(CV_PBC_path):
        CV_PBC_res.append(CV_PBC_path)
    else:
        CV_PBC_res.append("")

    CV_SVM_path = os.path.join(CV_DIR, "CV_SVM_%s.csv" % ds_name)
    if os.path.isfile(CV_SVM_path):
        CV_SVM_res.append(CV_SVM_path)
    else:
        CV_SVM_res.append("")

for i, (tr_pbc, cv_pbc, cv_svm) in enumerate(zip(tr_res, CV_PBC_res, CV_SVM_res)):
    fig = plt.figure(figsize=(32, 16), edgecolor="black", linewidth=5)
    fig_dir = os.path.join(RES_DIR, tr_pbc[0])
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    print("* %s - %s (Best LR: %s)" % (tr_pbc[0], tr_pbc[1], tr_pbc[2]))

    title = "%s (%s)" % (tr_pbc[1], tr_pbc[2])
    res_df = tr_pbc[3]
    df_epoch = res_df["Epoch"]
    df_L = res_df["Loss"]
    df_Etr = res_df["TrainingError"]
    df_Ets = res_df["TestingError"]

    ax_loss = fig.add_subplot(1, 2, 1)
    ax_loss.set_title("Loss: %s" % title)
    ax_trts = fig.add_subplot(1, 2, 2)
    ax_trts.set_title("Error: %s" % title)

    # CV
    one_tile = np.ones_like(df_epoch)
    res_cv_pbc = read_CV_result(cv_pbc)
    if len(res_cv_pbc) > 0:
        min_Lval_idx = np.argmin(res_cv_pbc[:, 2])
        min_pbc_Lval = res_cv_pbc[min_Lval_idx, 2]
        min_pbc_Ltr = res_cv_pbc[min_Lval_idx, 1]
        min_Lval_p = res_cv_pbc[min_Lval_idx, 0]

        print ("PBC Lval: %s (p=%s)" % (min_pbc_Lval, min_Lval_p))

        if show_numerials:
            ax_trts.plot(df_epoch, one_tile * min_pbc_Lval, color="darkviolet", lw=10,
                label="MPT+CV (p=%s, Lval=%04f)" % (min_Lval_p, min_pbc_Lval))
        else:
            ax_trts.plot(df_epoch, one_tile * min_pbc_Lval, color="darkviolet", lw=10,
                label="MPT+CV")

    res_cv_svm = read_CV_result(cv_svm)
    if len(res_cv_svm) > 0:
        min_Lval_idx = np.argmin(res_cv_svm[:, 3])
        min_svm_Lval = res_cv_svm[min_Lval_idx, 3]
        min_svm_Ltr = res_cv_svm[min_Lval_idx, 2]
        min_Lval_gamma = res_cv_svm[min_Lval_idx, 1]
        min_Lval_C = res_cv_svm[min_Lval_idx, 0]

        print ("SVM Lval: %s (C=%s, gamma=%s)" % (min_svm_Lval, min_Lval_C, min_Lval_gamma))

        if show_numerials:
            ax_trts.plot(df_epoch, one_tile * min_svm_Lval, color="blue", lw=10,
                label="SVM+CV (C=%s, gamma=%s, Lval=%04f)" % (min_Lval_C, min_Lval_gamma, min_svm_Lval))
        else:
            ax_trts.plot(df_epoch, one_tile * min_svm_Lval, color="blue", lw=10,
                label="SVM+CV")

    # Training
    print ("MBB Training: Epoch=%s Result; Etr: %s, Ets: %s (LR=%s, Loss=%s)"
           % (df_epoch.values[tr_pbc[4]], df_Etr.values[tr_pbc[4]], df_Ets.values[tr_pbc[4]], tr_pbc[2], df_L.values[tr_pbc[4]]))


    if show_numerials:
        ax_loss.plot(df_epoch, df_L, color="green", lw=10, label="Loss (%04f)" % df_L.values[tr_pbc[4]])
        ax_trts.plot(df_epoch, df_Etr, color="red", lw=10, label="MBB Training (%04f)" % df_Etr.values[tr_pbc[4]])
        ax_trts.plot(df_epoch, df_Ets, color="orange", lw=10, label="MBB Testing (%04f)" % df_Ets.values[tr_pbc[4]])
    else:
        ax_loss.plot(df_epoch, df_L, color="green", lw=10, label="Loss")
        ax_trts.plot(df_epoch, df_Etr, color="red", lw=10, label="MBB Training")
        ax_trts.plot(df_epoch, df_Ets, color="orange", lw=10, label="MBB Testing")

        #ax_trts.set_ylim(bottom=0.0)
        if tr_pbc[1] == "BreastCancer":
            ax_trts.set_ylim(top=0.10)
        elif tr_pbc[1] == "Abalone":
            ax_trts.set_ylim(top=0.60)
        elif tr_pbc[1] == "Cardiotocography2C":
            ax_trts.set_ylim(top=0.20)
        elif tr_pbc[1] == "German":
            ax_trts.set_ylim(top=0.55)
        elif tr_pbc[1] == "Satimage":
            ax_trts.set_ylim(top=0.20)
        elif tr_pbc[1] == "Spambase":
            ax_trts.set_ylim(top=0.16)
        elif tr_pbc[1] == "WineRed3C":
            ax_trts.set_ylim(top=0.70)
        elif tr_pbc[1] == "WineWhite3C":
            ax_trts.set_ylim(top=0.70)

    ax_loss.legend(loc="best")
    ax_trts.legend(loc="best")

    save_plt_fig(fig, os.path.join(fig_dir, "SERIES_%s.png" % tr_pbc[1]), close_plt_obj=True)

    """
    # for tex
    tex_res = []
    tex_res.append(["%s" % tr_pbc[1], "%s" % tr_pbc[2], "%.5f" % df_Etr.values[tr_pbc[4]], "%.5f" % df_Ets.values[tr_pbc[4]],
                    "%s" % int(min_Lval_p), "%.5f" % min_pbc_Ltr, "%.5f" % min_pbc_Lval,
                    "(%s, %s)" % (int(min_Lval_C), int(min_Lval_gamma)), "%.5f" % min_svm_Ltr, "%.5f" % min_svm_Lval])
    with open(os.path.join(RES_DIR, "tex_tablebody.txt"), "w") as f:
        for tr in tex_res:
            f.write("%s\n" % " & ".join(tr))
    """
