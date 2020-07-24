from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def entropy_gradients(optimizer, estimator, samples, var_list=None):
    surrogate_cost = tf.reduce_mean(
        tf.reduce_sum(tf.stop_gradient(-dlog_q) * samples, -1))
    grads_and_vars = optimizer.compute_gradients(
        surrogate_cost, var_list=var_list)
    return grads_and_vars


def entropy_surrogate(estimator, samples):
    dlog_q = estimator.compute_gradients(samples)
    surrogate = tf.reduce_mean(
        tf.reduce_sum(tf.stop_gradient(-dlog_q) * samples, -1))
    return surrogate

def kl_surrogate(estimator, q_data, p_data):
    entropy_sur = entropy_surrogate(estimator, q_data)

    cross_entropy_gradients = estimator.compute_gradients(p_data, q_data)
    cross_entropy_sur = tf.reduce_mean(tf.reduce_sum(
        tf.stop_gradient(cross_entropy_gradients) * q_data, -1))
    return -entropy_sur - cross_entropy_sur


def minimize_entropy(optimizer, estimator, samples, var_list=None):
    dlog_q = estimator.compute_gradients(samples)
    backprop_loss = -dlog_q / tf.cast(
        tf.reduce_prod(tf.shape(samples)[:-1]), dlog_q.dtype)
    opt_op = optimizer.minimize(
        samples, var_list=var_list, grad_loss=backprop_loss)
    return opt_op
