from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class ScoreEstimator(object):
    def __init__(self):
        pass

    def rbf_kernel(self, x1, x2, kernel_width):
        return tf.exp(-tf.reduce_sum(tf.square((x1 - x2) / kernel_width), axis=-1) / 2)

    def gram(self, x1, x2, kernel_width):
        x_row = tf.expand_dims(x1, -2)
        x_col = tf.expand_dims(x2, -3)
        return self.rbf_kernel(x_row, x_col, kernel_width)

    def grad_gram(self, x1, x2, kernel_width):
        x_row = tf.expand_dims(x1, -2)
        x_col = tf.expand_dims(x2, -3)
        G = self.rbf_kernel(x_row, x_col, kernel_width)
        diff = (x_row - x_col) / (kernel_width ** 2)
        G_expand = tf.expand_dims(G, axis=-1)
        grad_x2 = G_expand * diff
        grad_x1 = G_expand * (-diff)
        return G, grad_x1, grad_x2

    def heuristic_kernel_width(self, x_samples, x_basis):
        x_dim = tf.shape(x_samples)[-1]
        n_samples = tf.shape(x_samples)[-2]
        n_basis = tf.shape(x_basis)[-2]
        x_samples_expand = tf.expand_dims(x_samples, -2)
        x_basis_expand = tf.expand_dims(x_basis, -3)
        pairwise_dist = tf.abs(x_samples_expand - x_basis_expand)

        length = len(pairwise_dist.get_shape())
        reshape_dims = list(range(length-3)) + [length-1, length-3, length-2]
        pairwise_dist = tf.transpose(pairwise_dist, reshape_dims)

        k = n_samples * n_basis // 2
        top_k_values = tf.nn.top_k(
            tf.reshape(pairwise_dist, [-1, x_dim, n_samples * n_basis]),
            k=k).values

        kernel_width = tf.reshape(top_k_values[:, :, -1],
                                  tf.concat([tf.shape(x_samples)[:-2], [1, 1, x_dim]], axis=0))
        kernel_width = kernel_width * (tf.to_float(x_dim) ** 0.5)
        kernel_width = kernel_width + tf.to_float(kernel_width < 1e-6) * 1.
        return tf.stop_gradient(kernel_width)

    def compute_gradients(self, samples, x=None):
        raise NotImplementedError()