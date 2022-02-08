# coding: utf-8

import os
import time
# import tensorflow as tf
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class BaseTrain:
    def __init__(self, classifier, trainer, optimizer, sess, conf):
        self.classifier = classifier
        self.trainer = trainer
        self.optimizer = optimizer
        self.sess = sess
        self.conf = conf
        self.result = []

        self.sess.run(tf.global_variables_initializer())


    def train(self):
        learning_start = time.time()
        for cur_epoch in range(0, self.conf.epoch+1, 1):
            self.train_epoch(cur_epoch)
        time_result = time.time() - learning_start

        self.show_result_series()
        self.output_result()
        return time_result

        
    def training_epoch(self, current_epoch):
        # for batch loop
        raise NotImplementedError

    def training_step(self):
        # for current data training
        raise NotImplementedError

    def show_result_series(self):
        # for current data training
        raise NotImplementedError

    def output_result(self):
        df_res = pd.DataFrame.from_dict(self.result)
        df_res.to_csv(os.path.join(self.conf.output, "result.csv"), index=False)
        cf_res = pd.DataFrame.from_dict(self.conf.items())
        cf_res.to_csv(os.path.join(self.conf.output, "config.csv"), index=False, header=False)


class ClassifierModel():
    def __init__(self, conf):
        self.conf = conf
        self.x = None
        self.y = None
        self.dim_num = None
        self.class_num = None
        self.proto_num = None
        self.A = None
        self.proto = None
        self.nearest_proto = None
        self.nearest_proto_ind = None
        self.class_i_ind = None
        self.class_j_max_ind = None
        self.d = None
        self.D = None
        self.diff_d_by_x = None
        self.diff_d_by_x_norm = None
        self.hm = None
        self.diff_hm = None
        self.diff_hm_norm = None
        self.y_pred = None
        self.y_true = None
        self.error = None

    def build_model(self):
        raise NotImplementedError


class TrainingModel():
    def __init__(self, classifier):
        self.classifier = classifier
        self.L = None
        self.mean_diff_l_by_A = None

    def build_model(self):
        raise NotImplementedError


class OptimizerModel():
    def __init__(self, classifier, trainer):
        self.classifier = classifier
        self.trainer = trainer
        self.epoch = None
        self.epsilon = None
        self.A_next = None

    def build_model(self):
        raise NotImplementedError
