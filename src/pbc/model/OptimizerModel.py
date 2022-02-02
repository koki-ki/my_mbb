# coding: utf-8
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from pbc.base.base import OptimizerModel

class GDOptimizer(OptimizerModel):
    def __init__(self, classifier, trainer):
        super(GDOptimizer, self).__init__(classifier, trainer)
        self.build_model()

    def build_model(self):
        self.epoch = tf.placeholder(shape=None, dtype=tf.float32)
        self.epsilon = tf.placeholder(shape=None, dtype=tf.float32)

        # c * p * d
        delta_A = tf.multiply(self.epsilon, self.trainer.mean_diff_l_by_A)
        self.A_next = tf.assign(self.classifier.A, tf.subtract(self.classifier.A, delta_A))

