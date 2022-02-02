#
# coding: utf-8
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from pbc.base.base import TrainingModel


# Inf: split some value by 0 on np
# NaN: split 0 by 0 on np

class MaximumBayesBoundarynessTraining(TrainingModel):
    def __init__(self, classifier):
        super(MaximumBayesBoundarynessTraining, self).__init__(classifier)
        self.build_model()

    def build_model(self):
        tf_pi = tf.constant(np.pi, dtype=tf.float32)
        tf_float_max = tf.constant(np.finfo("float32").max, dtype=tf.float32)
        tf_float_eps = tf.constant(10**(-15), dtype=tf.float32)

        self.y_seg_map = tf.placeholder(shape=[None, None], dtype=tf.float32)  # seg * S; NaN
        self.h_seg_map = tf.placeholder(shape=[None, None], dtype=tf.float32)  # seg * S; Nan
        self.h_S = tf.placeholder(shape=[None, 1], dtype=tf.float32)  # S * 1
        self.g_seg_ind = tf.placeholder(shape=[None, 2], dtype=tf.float32)  # seg * 2
        self.S_ind = tf.placeholder(shape=[None, 1], dtype=tf.int32)  # S * 1
        S_ind_flatten = tf.squeeze(self.S_ind, axis=1)

        normed_h_seg_map = tf.where(tf.is_nan(self.h_seg_map),
                                    tf.zeros_like(self.h_seg_map),
                                    self.h_seg_map)  # seg * S; zero-

        y_seg_map_mask = tf.where(tf.is_nan(self.y_seg_map),
                                  tf.zeros_like(self.y_seg_map),
                                  tf.ones_like(self.y_seg_map))  # seg * S; zero-one
        self.S_cnt = tf.reduce_sum(y_seg_map_mask, axis=0, keepdims=True)  # 1 * S ## the size of segment by samples
        
        
        y_S = tf.nn.embedding_lookup(self.classifier.y, S_ind_flatten)  # S * c
        y_ast_S = tf.nn.embedding_lookup(self.classifier.y_ast, S_ind_flatten)  # S * c
        self.dy_S = tf.nn.embedding_lookup(self.classifier.dy, S_ind_flatten)  # S * 1

        # S * 1　
        k = tf.multiply(tf.divide(1.0, tf.sqrt(tf.multiply(2.0, tf_pi))),
                        tf.exp(tf.multiply(-0.5, tf.square(tf.subtract(self.dy_S, 0.0) / self.h_S))))
        k = tf.where(tf.is_nan(k),
                     tf.multiply(tf.ones_like(k), tf.multiply(-1.0, tf_float_max)),
                     k)
        k = tf.where(tf.is_finite(k),
                     k,
                     tf.multiply(tf.sign(k), tf_float_max))
        size_of_samples = tf.cast(k.shape[1], tf.float32)
        # tf.to_float(size_of_samples, name='ToFloat')


        # 1
        sum_k = tf.reduce_sum(k, axis=0)
        sum_k = tf.where(tf.is_nan(sum_k),
                         tf.zeros_like(sum_k),
                         sum_k)
        sum_k = tf.where(tf.is_finite(sum_k),
                         sum_k,
                         tf.multiply(tf.sign(sum_k), tf_float_max))
        
        # S * 1
        self.W = tf.divide(k, sum_k)
        self.W = tf.where(tf.is_nan(self.W),
                         tf.zeros_like(self.W),
                         self.W)
        self.W = tf.where(tf.is_finite(self.W),
                         self.W,
                         tf.multiply(tf.sign(self.W), tf_float_max))
        # self_S_cnt = tf.transpose(self.S_cnt)  #S*1                       
        self.W = tf.ones_like(self.W)
        self.W = tf.divide(self.W, sum_k)

        # seg * 1, seg * 1
        g_i_seg_ind, g_j_seg_ind = tf.split(self.g_seg_ind, 2, axis=1)

        # seg * S
        d_i_seg_mask = tf.where(tf.equal(self.y_seg_map, g_i_seg_ind)
                                & tf.equal(tf.is_nan(self.y_seg_map), False),
                                tf.ones_like(self.y_seg_map),
                                tf.zeros_like(self.y_seg_map))
        # seg * S
        d_j_seg_mask = tf.where(tf.equal(self.y_seg_map, g_j_seg_ind)
                                & tf.equal(tf.is_nan(self.y_seg_map), False),
                                tf.ones_like(self.y_seg_map),
                                tf.zeros_like(self.y_seg_map))

        # seg * S
        p_i_seg_map = tf.where(tf.equal(d_i_seg_mask, 0.0),
                               tf.zeros_like(d_i_seg_mask),
                               tf.multiply(tf.divide(1.0, tf.sqrt(tf.multiply(2.0, tf_pi))),
                                           tf.exp(tf.multiply(-1.0,
                                                  tf.divide(tf.square(tf.multiply(d_i_seg_mask, tf.transpose(self.dy_S))),
                                                            tf.multiply(2.0, tf.square(tf.multiply(d_i_seg_mask, normed_h_seg_map))))))))
        p_i_seg_map = tf.where(tf.is_nan(p_i_seg_map),
                               tf.zeros_like(p_i_seg_map),
                               p_i_seg_map)

        # seg * 1
        p_i_seg = tf.reduce_sum(p_i_seg_map, axis=1, keepdims=True)
        p_i_seg = tf.add(p_i_seg, tf_float_eps)

        # seg * S
        p_j_seg_map = tf.where(tf.equal(d_j_seg_mask, 0.0),
                               tf.zeros_like(d_j_seg_mask),
                               tf.multiply(tf.divide(1.0, tf.sqrt(tf.multiply(2.0, tf_pi))),
                                           tf.exp(tf.multiply(-1.0,
                                                  tf.divide(tf.square(tf.multiply(d_j_seg_mask, tf.transpose(self.dy_S))),
                                                            tf.multiply(2.0, tf.square(tf.multiply(d_j_seg_mask, normed_h_seg_map))))))))
        p_j_seg_map = tf.where(tf.is_nan(p_j_seg_map),
                               tf.zeros_like(p_j_seg_map),
                               p_j_seg_map)

        # seg * 1
        p_j_seg = tf.reduce_sum(p_j_seg_map, axis=1, keepdims=True)
        p_j_seg = tf.add(p_j_seg, tf_float_eps)

        # seg * 1
        P_i_seg = tf.divide(p_i_seg, tf.add(p_i_seg, p_j_seg))
        P_i_seg = tf.where(tf.is_nan(P_i_seg),
                           tf.zeros_like(P_i_seg),
                           P_i_seg)
        P_i_seg = tf.clip_by_value(P_i_seg, 0.0, 1.0)

        # seg * 1
        P_j_seg = tf.divide(p_j_seg, tf.add(p_j_seg, p_i_seg))
        P_j_seg = tf.where(tf.is_nan(P_j_seg),
                           tf.zeros_like(P_j_seg),
                           P_j_seg)
        P_j_seg = tf.clip_by_value(P_j_seg, 0.0, 1.0)

        # seg * S
        P_seg_map = tf.add(tf.multiply(d_i_seg_mask, P_i_seg), tf.multiply(d_j_seg_mask, P_j_seg))

        # 1 * S -> S * 1
        self.P_hat = tf.transpose(tf.divide(tf.reduce_sum(P_seg_map, axis=0, keepdims=True), self.S_cnt))
        self.P_hat = tf.where(tf.is_nan(self.P_hat),
                           tf.zeros_like(self.P_hat),
                           self.P_hat)
        self.P_hat = tf.clip_by_value(self.P_hat, 0.0, 1.0)

        # S * 1
        H_hat = tf.multiply(tf.divide(1.0, tf.log(2.0)),
                            tf.add(tf.multiply(-1.0, tf.multiply(self.P_hat, tf.log(self.P_hat))),
                                   tf.multiply(-1.0, tf.multiply(tf.subtract(1.0, self.P_hat), tf.log(tf.subtract(1.0, self.P_hat))))))

        H_hat = tf.where(tf.is_finite(H_hat), 
                         H_hat, 
                        tf.zeros_like(H_hat))

        # S * 1
        U_hat = tf.subtract(1.0, H_hat)

        # S * 1
        l = U_hat

        # 1
        #self.L = tf.reduce_mean(tf.reshape(l, [-1]), axis=0)
        self.L = tf.reduce_sum(tf.reshape(l, [-1]), axis=0)


        ### differential
        # S * d
        x_S = tf.nn.embedding_lookup(self.classifier.x, S_ind_flatten)
        # S * c * p * d
        diff_g_by_A = tf.map_fn(lambda x: tf.add(tf.multiply(-1.0, self.classifier.A), x), x_S)
        # S * c * p
        nearest_proto_ind_S = tf.nn.embedding_lookup(self.classifier.nearest_proto_ind, S_ind_flatten)
        # S * c
        diff_dy_by_g = tf.add(tf.multiply(-1.0, y_S), tf.multiply(1.0, y_ast_S))

        ## second term (multiply W)
        # S * 1
        diff_Uhat_by_Hhat = tf.multiply(-1.0, tf.ones_like(H_hat))

        # S * 1
        diff_Hhat_by_Phat = tf.multiply(tf.divide(-1.0, tf.log(2.0)),
                                        tf.log(tf.divide(self.P_hat, tf.add(tf.subtract(1.0, self.P_hat), tf_float_eps))))

        diff_Hhat_by_Phat = tf.where(tf.is_nan(diff_Hhat_by_Phat),
                                     tf.zeros_like(diff_Hhat_by_Phat),
                                     diff_Hhat_by_Phat)

        # diff_Phat_by_A
        # seg * 1
        diff_P_i_seg = tf.divide(p_j_seg, tf.square(tf.add(p_i_seg, p_j_seg)))

        diff_P_i_seg = tf.where(tf.is_nan(diff_P_i_seg),
                                tf.zeros_like(diff_P_i_seg),
                                diff_P_i_seg)
        # seg * 1
        diff_P_j_seg = tf.divide(p_i_seg, tf.square(tf.add(p_j_seg, p_i_seg)))
        diff_P_j_seg = tf.where(tf.is_nan(diff_P_j_seg),
                                tf.zeros_like(diff_P_j_seg),
                                diff_P_j_seg)
        # seg * 1
        #diff_P_i_seg_oppo = tf.divide(tf.multiply(-1.0, p_i_seg), tf.square(tf.add(p_i_seg, p_j_seg)))
        #diff_P_i_seg_oppo = tf.where(tf.is_nan(diff_P_i_seg_oppo),
        #                        tf.zeros_like(diff_P_i_seg_oppo),
        #                        diff_P_i_seg_oppo)
        # seg * 1
        #diff_P_j_seg_oppo = tf.divide(tf.multiply(-1.0, p_j_seg), tf.square(tf.add(p_j_seg, p_i_seg)))
        #diff_P_j_seg_oppo = tf.where(tf.is_nan(diff_P_j_seg_oppo),
        #                        tf.zeros_like(diff_P_j_seg_oppo),
        #                        diff_P_j_seg_oppo)
        # seg * S
        diff_p_i_seg_map = tf.where(tf.equal(d_i_seg_mask, 0.0),
                                    tf.zeros_like(d_i_seg_mask),
                                    tf.multiply(tf.multiply(-1.0,
                                        tf.divide(tf.multiply(d_i_seg_mask, tf.transpose(self.dy_S)),
                                                  tf.multiply(tf.sqrt(tf.multiply(2.0, tf_pi)),
                                                              tf.square(tf.multiply(d_i_seg_mask, normed_h_seg_map))))),
                                        tf.exp(tf.multiply(-1.0,
                                               tf.divide(tf.square(tf.multiply(d_i_seg_mask, tf.transpose(self.dy_S))),
                                                         tf.multiply(2.0, tf.square(tf.multiply(d_i_seg_mask, normed_h_seg_map))))))))
        diff_p_i_seg_map = tf.where(tf.is_nan(diff_p_i_seg_map),
                                    tf.zeros_like(diff_p_i_seg_map),
                                    diff_p_i_seg_map)
        # seg * 1
        #diff_p_i_seg = tf.reduce_sum(diff_p_i_seg_map, axis=1, keepdims=True)

        # seg * S
        diff_p_j_seg_map = tf.where(tf.equal(d_j_seg_mask, 0.0),
                                    tf.zeros_like(d_j_seg_mask),
                                    tf.multiply(tf.multiply(-1.0,
                                        tf.divide(tf.multiply(d_j_seg_mask, tf.transpose(self.dy_S)),
                                                  tf.multiply(tf.sqrt(tf.multiply(2.0, tf_pi)),
                                                              tf.square(tf.multiply(d_j_seg_mask, normed_h_seg_map))))),
                                        tf.exp(tf.multiply(-1.0,
                                               tf.divide(tf.square(tf.multiply(d_j_seg_mask, tf.transpose(self.dy_S))),
                                                         tf.multiply(2.0, tf.square(tf.multiply(d_j_seg_mask, normed_h_seg_map))))))))
        diff_p_j_seg_map = tf.where(tf.is_nan(diff_p_j_seg_map),
                                    tf.zeros_like(diff_p_j_seg_map),
                                    diff_p_j_seg_map)
        # seg * 1
        #diff_p_j_seg = tf.reduce_sum(diff_p_j_seg_map, axis=1, keepdims=True)

        # S * 1
        diff_Phat_by_dy = tf.transpose(tf.divide(
                              tf.reduce_sum(
                                  tf.multiply(y_seg_map_mask,
                                      tf.add(
                                          tf.multiply(d_i_seg_mask, tf.multiply(diff_p_i_seg_map, diff_P_i_seg)),
                                          tf.multiply(d_j_seg_mask, tf.multiply(diff_p_j_seg_map, diff_P_j_seg)))),
                              axis=0, keepdims=True),
                          self.S_cnt))

        #diff_Phaty_by_dy = tf.transpose(tf.divide(
        #                       tf.reduce_sum(tf.add(
        #                           tf.multiply(d_i_seg_mask, tf.multiply(diff_P_i_seg, diff_p_i_seg)),
        #                           tf.multiply(d_j_seg_mask, tf.multiply(diff_P_j_seg, diff_p_j_seg))),
        #                       axis=0, keepdims=True),
        #                   self.S_cnt))
        #diff_Phaty_by_dy_ast = tf.transpose(tf.divide(
        #                       tf.reduce_sum(tf.add(
        #                           tf.multiply(d_j_seg_mask, tf.multiply(diff_P_i_seg_oppo, diff_p_j_seg)),
        #                           tf.multiply(d_i_seg_mask, tf.multiply(diff_P_j_seg_oppo, diff_p_i_seg))),
        #                       axis=0, keepdims=True),
        #                   self.S_cnt))

        # S * 1
        diff_Uhat_by_Phat = tf.multiply(tf.multiply(diff_Uhat_by_Hhat, diff_Hhat_by_Phat), self.W)
        # S * 1
        diff_Uhat_by_d = tf.multiply(diff_Uhat_by_Phat, diff_Phat_by_dy)
        # S * c -> S * c * 1
        diff_Uhat_by_g = tf.expand_dims(tf.multiply(diff_Uhat_by_d, diff_dy_by_g), axis=2)
        # S * c * p -> S * c * p * 1
        diff_Uhat_by_g_proto = tf.expand_dims(tf.multiply(nearest_proto_ind_S, diff_Uhat_by_g), axis=3)
        # S * c * p * d
        second_term = tf.multiply(diff_g_by_A, diff_Uhat_by_g_proto)
        second_term /= size_of_samples 
        # S * c * p * d
        diff_l_by_A = second_term
        diff_l_by_A = tf.where(tf.is_nan(diff_l_by_A),
                                tf.zeros_like(diff_l_by_A),
                                diff_l_by_A)
        diff_l_by_A = tf.where(tf.is_finite(diff_l_by_A),
                                diff_l_by_A,
                                tf.multiply(tf.sign(diff_l_by_A), tf_float_max))
        # c * p * d
        self.mean_diff_l_by_A = tf.reduce_sum(diff_l_by_A, axis=0)



