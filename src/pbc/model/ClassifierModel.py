# coding: utf-8
import numpy as np
# import tensorflow as tf
from sklearn.cluster import KMeans

from pbc.base.base import ClassifierModel

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class MultiClassKmeansPrototypeClassifier(ClassifierModel):
    def __init__(self, conf, X, y):
        super(MultiClassKmeansPrototypeClassifier, self).__init__(conf)
        self.build_model(X, y)

    def build_model(self, X, y):
        # Input parameter
        self.dim_num = X.shape[1]  # # of Dimension d
        self.class_num = y.shape[1]  # # of Class c (One-hot vector)
        self.proto_num = self.conf.proto  # # of Proto p
        self.x = tf.placeholder(shape=[None, self.dim_num], dtype=tf.float32)  # n * d
        self.y = tf.placeholder(shape=[None, self.class_num], dtype=tf.float32)  # n * c

        # kmeans clustering
        kmeans = KMeans(n_clusters=self.proto_num, n_init=20, random_state=0)  # p * d
        kmeans_proto = np.array([kmeans.fit(X[np.where(np.argmax(y, axis=1)==cls)]).cluster_centers_ for cls in range(self.class_num)])  # c * p * d

        # PBC model parameter \Lambda
        self.A = tf.Variable(kmeans_proto, dtype=tf.float32)  # c * p * d

        # Discriminant Function: g = (X^T, A) - 0.5(A^T, A) (inner product)
        g = tf.reduce_sum(tf.map_fn(lambda x: tf.subtract(tf.multiply(self.A, x),
                                    tf.multiply(0.5, tf.square(self.A))), self.x), axis=3)  # n * c * p

        # Decision rule (Maximum g by each class)
        self.nearest_proto = tf.reduce_max(g, axis=2)  # n * c
        self.nearest_proto_ind = tf.one_hot(tf.argmax(g, axis=2), self.proto_num)  # n * c * p

        # Correct class g
        not_y_value = tf.multiply(tf.ones_like(self.y), -np.inf)  # n * c
        y_value = tf.multiply(self.y, self.nearest_proto)  # n * c
        g_y_oh = tf.where(tf.equal(y_value, 0), not_y_value, y_value)  # n * c
        g_y = tf.reduce_max(g_y_oh, axis=1, keepdims=True)  # n * 1

        # Best incorrect class g
        y_j = tf.subtract(tf.ones_like(self.y), self.y)  # n * c
        not_yast_value = tf.multiply(tf.ones_like(y_j), -np.inf)  # n * c
        yast_value = tf.multiply(y_j, self.nearest_proto)  # n * c
        g_yast_oh = tf.where(tf.equal(yast_value, 0), not_yast_value, yast_value)  # n * c
        g_yast = tf.reduce_max(g_yast_oh, axis=1, keepdims=True)  # n * 1
        self.y_ast = tf.one_hot(tf.argmax(g_yast_oh, axis=1), self.class_num)  # n * c

        # 2 class g
        self.gs = tf.add(tf.multiply(self.y, g_y), tf.multiply(self.y_ast, g_yast))  # n * c
        self.gs_ind_oh = tf.add(self.y, self.y_ast)  # n * c
        self.gs_ind = tf.concat([tf.reshape(tf.argmax(self.y, axis=1), [-1, 1]),
                                 tf.reshape(tf.argmax(self.y_ast, axis=1), [-1, 1])], axis=1)  # n * 2

        # Mis-classification measure
        self.dy = tf.subtract(g_yast, g_y)  # n * 1

        # Error rate
        self.y_pred = tf.argmax(self.nearest_proto, axis=1)  # n
        self.y_true = tf.argmax(self.y, axis=1)  # n
        accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_pred, self.y_true), tf.float32))  # 1
        self.error = tf.subtract(1.0, accuracy)  # 1

       