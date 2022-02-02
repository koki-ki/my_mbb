# config: utf-8
import os
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = os.path.abspath(os.path.join(
    os.path.abspath(__file__), "../../../dataset/HO"))


def output_txt(output_path, message_list):  # [message1, ..]
    with open(output_path, "w") as f:
        for ml in message_list:
            print(ml)
            f.write("%s\n" % ml)


def load_data(dataset_path, delimiter=',', training=0.5, testing=0.5, output_dir=OUTPUT_DIR,
              save_split_data=True, load_split_data=True, normalize_by_z_score=True, make_one_hot_label=True, only_training_set=False, overwrite=False, randomize=False, seed=0):

    print(OUTPUT_DIR)

    # Confirm path exist
    print(" * Dataset path: %s" % dataset_path)
    if not os.path.exists(os.path.dirname(dataset_path)):
        print(" *** load_data: ERROR; Cannot find dataset dir.")
        print(os.path.dirname(dataset_path))
        quit(1)

    # HO splitting ratio
    if (training + testing) > 1.0:
        training = int(training)
        testing = int(testing)

    # Confirm Tr/Ts ratio
    training_sample_ratio = training / (training + testing)
    testing_sample_ratio = testing / (training + testing)
    if training_sample_ratio == 1.0:
        only_training_set = True

    # Setting save path
    trn = str(training).replace(".", "p")
    tsn = str(testing).replace(".", "p")

    base_name = os.path.basename(dataset_path).split(".")[0]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.join(
        output_dir, "%s_training-%s_testing-%s.csv" % (base_name, trn, tsn))

    # Read existing data (if you already made HO splitting data)
    if load_split_data:
        if overwrite:
            print(" *** Overwrite=True; Create new dataset")
        else:
            if not os.path.exists(save_path):
                print(
                    " *** WARNING: %s could not find in the path, Create new dataset" % save_path)
            else:
                # Load stored data
                print("  * Load stored data from %s" % save_path)
                exist_data = np.genfromtxt(
                    save_path, delimiter=delimiter, filling_values=0)
                x_data = exist_data[:, :-1].astype(float)
                y_class = exist_data[:, -1].astype(int)
                if make_one_hot_label:
                    # One-hot vector (max_value + 1 (for "0" class))
                    one_hot = np.identity(max(y_class) + 1)[y_class]
                    y_class = np.array(one_hot)
                trp = int(exist_data.shape[0] * training_sample_ratio)
                tsp = int(exist_data.shape[0] * testing_sample_ratio)
                return x_data[:trp], y_class[:trp], x_data[trp:trp+tsp], y_class[trp:trp+tsp]

    # Create new dataset
    # data = np.genfromtxt(dataset_path, delimiter=delimiter, filling_values=0)
    data = np.genfromtxt(dataset_path, delimiter=delimiter, filling_values=0)
    x_data = data[:, :-1]
    y_class = data[:, -1]

    # print(data)
    # print(x_data)
    # print(y_class)
    # print(np.isnan(data))
    # print(np.count_nonzero(np.isnan(data)))
    # print(list(zip(*np.where(np.isnan(data)))))

    # Split index into training and testing
    tri, tsi = [], []
    if randomize:
        # Random splitting
        np.random.seed(seed)

        tr_index = []
        ts_index = []
        y_cat = np.unique(y_class, return_counts=False)
        for yc in y_cat:
            yc_idx = np.where(y_class == yc)[0]
            tr_idx = np.random.choice(yc_idx, size=int(
                len(yc_idx) * training_sample_ratio), replace=False)
            ts_idx = list(set(yc_idx) - set(tr_idx))
            tr_index.extend(tr_idx)
            ts_index.extend(ts_idx)
        tri = np.random.choice(tr_index, size=len(tr_index), replace=False)
        if not only_training_set:
            tsi = np.random.choice(ts_index, size=len(ts_index), replace=False)
    else:
        # Sequential splitting
        trp = int(data.shape[0] * training_sample_ratio)
        tsp = int(data.shape[0] * testing_sample_ratio)
        all_index = list(range(data.shape[0]))
        tri = all_index[:trp]
        tsi = all_index[trp:trp+tsp]

    # Generalize y label (Start y index from 0)
    raw_labels = sorted(set(y_class))
    mapping = {}
    for i, raw_label in enumerate(raw_labels):
        mapping[raw_label] = i
    for i in range(len(y_class)):
        y_old = y_class[i]
        y_class[i] = mapping[y_old]

    # z score normalization
    if normalize_by_z_score:
        print(" * Normalize by z score: %s" % normalize_by_z_score)
        sc = StandardScaler()
        sc.fit(x_data)
        x_data = np.array(sc.transform(x_data))

    # Save new data
    if save_split_data:
        messages = []
        messages.append(" * Create date: %s" %
                        datetime.datetime.today().strftime("%Y-%m-%d_%H:%M:%S"))
        messages.append(" * Save dataset to %s" % save_path)
        messages.append("  - Seed: %s" % seed)
        messages.append("  - Dimension: %s" % x_data.shape[1])
        y, N_y = np.unique(y_class, return_counts=True)
        messages.append("  - Class: %s" % len(y))
        messages.append("  - Total Samples: %s" % np.sum(N_y))
        for i in range(len(y)):
            R_y = N_y[i] / np.sum(N_y)
            messages.append("   - Class %s: %s (class ratio: %s)" %
                            (y[i], N_y[i], R_y))
        y_tr, N_y_tr = np.unique(y_class[tri], return_counts=True)
        S_y_tr = np.sum(N_y_tr) / len(y_class)
        messages.append("  - Training Samples: %s (sample ratio: %s)" %
                        (np.sum(N_y_tr), S_y_tr))
        for i in range(len(y_tr)):
            R_y_tr = N_y_tr[i] / np.sum(N_y_tr)
            messages.append("   - Class %s: %s (class ratio: %s)" %
                            (y_tr[i], N_y_tr[i], R_y_tr))
        y_ts, N_y_ts = np.unique(y_class[tsi], return_counts=True)
        S_y_ts = np.sum(N_y_ts) / len(y_class)
        messages.append("  - Testing Samples: %s (sample ratio: %s)" %
                        (np.sum(N_y_ts), S_y_ts))
        for i in range(len(y_ts)):
            R_y_ts = N_y_ts[i] / np.sum(N_y_ts)
            messages.append("   - Class %s: %s (class ratio: %s)" %
                            (y_ts[i], N_y_ts[i], R_y_ts))
        train_data = np.hstack(
            (x_data[tri], np.reshape(y_class[tri], (-1, 1))))
        test_data = np.hstack((x_data[tsi], np.reshape(y_class[tsi], (-1, 1))))
        new_data = np.vstack((train_data, test_data))
        np.savetxt(save_path, new_data, delimiter=delimiter)
        output_txt(os.path.join(
            output_dir, "log_%s_training-%s_testing-%s.txt" % (base_name, trn, tsn)), messages)

    # Make one-hot vector in y
    y_class = np.array(y_class, dtype=np.int)
    if make_one_hot_label:
        print(" * Label one-hot expression: %s" % make_one_hot_label)
        # One-hot vector (max_value + 1 (for "0" class))
        one_hot = np.identity(max(y_class) + 1)[y_class]
        y_class = np.array(one_hot)

    return x_data[tri], y_class[tri], x_data[tsi], y_class[tsi]
