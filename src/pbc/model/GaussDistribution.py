import numpy as np
import tensorflow as tf

def GaussA(sigma):
    tf_pi = tf.constant(np.pi, dtype=tf.float32)
    # y = tf.sqrt(tf.multiply(tf.multiply(tf.constant(2.0), tf.constant(np.pi)), tf.square(sigma)))
    # a = tf.divide(tf.constant(1.0), y)
    # a = tf.divide(
    #         tf.sqrt(
    #             tf.multiply(tf.constant(2.0), 
    #                         tf.multiply(tf_pi, tf.square(sigma)))
    #         ), 
    #         tf.constant(2.0)
    #     )
    a = tf.sqrt(
                tf.multiply(tf.constant(2.0), 
                            tf.multiply(tf_pi, tf.square(sigma)))
        )
    return a



