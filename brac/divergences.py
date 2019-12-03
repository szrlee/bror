# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Divergences for BRAC agents."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf
from behavior_regularized_offline_rl.brac import utils

EPS = 1e-8  # Epsilon for avoiding numerical issues.
CLIP_EPS = 1e-3  # Epsilon for clipping actions.


@gin.configurable
def gradient_penalty(s, a_p, a_b, c_fn, gamma=5.0):
  """Calculates interpolated gradient penalty."""
  batch_size = s.shape[0]
  alpha = tf.random.uniform([batch_size])
  a_intpl = a_p + alpha[:, None] * (a_b - a_p)
  with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(a_intpl)
    c_intpl = c_fn(s, a_intpl)
  grad = tape.gradient(c_intpl, a_intpl)
  slope = tf.sqrt(EPS + tf.reduce_sum(tf.square(grad), axis=-1))
  grad_penalty = tf.reduce_mean(tf.square(tf.maximum(slope - 1.0, 0.0)))
  return grad_penalty * gamma


class Divergence(object):
  """Basic interface for divergence."""

  def dual_estimate(self, s, a_p, a_b, c_fn):
    raise NotImplementedError

  def dual_critic_loss(self, s, a_p, a_b, c_fn):
    return (- tf.reduce_mean(self.dual_estimate(s, a_p, a_b, c_fn))
            + gradient_penalty(s, a_p, a_b, c_fn))

  def primal_estimate(self, s, p_fn, b_fn, n_samples, action_spec=None):
    raise NotImplementedError


class FDivergence(Divergence):
  """Interface for f-divergence."""

  def dual_estimate(self, s, a_p, a_b, c_fn):
    logits_p = c_fn(s, a_p)
    logits_b = c_fn(s, a_b)
    return self._dual_estimate_with_logits(logits_p, logits_b)

  def _dual_estimate_with_logits(self, logits_p, logits_b):
    raise NotImplementedError

  def primal_estimate(self, s, p_fn, b_fn, n_samples, action_spec=None):
    _, apn, apn_logp = p_fn.sample_n(s, n_samples)
    _, abn, abn_logb = b_fn.sample_n(s, n_samples)
    # Clip actions here to avoid numerical issues.
    apn_logb = b_fn.get_log_density(
        s, utils.clip_by_eps(apn, action_spec, CLIP_EPS))
    abn_logp = p_fn.get_log_density(
        s, utils.clip_by_eps(abn, action_spec, CLIP_EPS))
    #tf.print("abn_logp", abn_logp)
    #tf.print("abn_logb", abn_logb)
    return self._primal_estimate_with_densities(
        apn_logp, apn_logb, abn_logp, abn_logb)

  def primal_estimate_internal(self, s, p_fn, abn, abn_logb, n_samples, action_spec=None):
    batch_size = s.shape[0]
    abn = tf.reshape(abn, [n_samples, batch_size, -1])
    abn_logb = tf.reshape(abn_logb, [n_samples, batch_size])
    # Clip actions here to avoid numerical issues.
    abn_logp = p_fn.get_log_density(
        s, utils.clip_by_eps(abn, action_spec, CLIP_EPS))
    IS_ratio = tf.exp(abn_logp - abn_logb)
    #tf.print("IS_ratio", IS_ratio)
    IS_ratio = tf.minimum(1.0, IS_ratio)
    #tf.print("abn_logp", abn_logp)
    #tf.print("abn_logp - abn_logb", abn_logp - abn_logb)
    return tf.reduce_mean(tf.multiply(IS_ratio, abn_logp - abn_logb), axis=0)
    #return tf.reduce_mean(abn_logp - abn_logb, axis=0)

  def _primal_estimate_with_densities(
      self, apn_logp, apn_logb, abn_logp, abn_logb):
    raise NotImplementedError


class KL(FDivergence):
  """KL divergence."""

  def _dual_estimate_with_logits(self, logits_p, logits_b):
    return (- utils.soft_relu(logits_b)
            + tf.log(utils.soft_relu(logits_p) + EPS) + 1.0)

  def _primal_estimate_with_densities(
      self, apn_logp, apn_logb, abn_logp, abn_logb):
    return tf.reduce_mean(apn_logp - apn_logb, axis=0)


class W(FDivergence):
  """Wasserstein distance."""

  def _dual_estimate_with_logits(self, logits_p, logits_b):
    return logits_p - logits_b


@gin.configurable
def laplacian_kernel(x1, x2, sigma=20.0):
  d12 = tf.reduce_sum(
      tf.abs(x1[None] - x2[:, None]), axis=-1)
  k12 = tf.exp(- d12 / sigma)
  return k12


@gin.configurable
def mmd(x1, x2, kernel, use_sqrt=False):
  k11 = tf.reduce_mean(kernel(x1, x1), axis=[0, 1])
  k12 = tf.reduce_mean(kernel(x1, x2), axis=[0, 1])
  k22 = tf.reduce_mean(kernel(x2, x2), axis=[0, 1])
  if use_sqrt:
    return tf.sqrt(k11 + k22 - 2 * k12 + EPS)
  else:
    return k11 + k22 - 2 * k12


class MMD(Divergence):
  """MMD."""

  def primal_estimate(
      self, s, p_fn, b_fn, n_samples,
      kernel=laplacian_kernel, action_spec=None):
    apn = p_fn.sample_n(s, n_samples)[1]
    abn = b_fn.sample_n(s, n_samples)[1]
    return mmd(apn, abn, kernel)

  def primal_estimate_internal(
      self, s, p_fn, abn, abn_logb, n_samples,
      kernel=laplacian_kernel, action_spec=None):
    batch_size = s.shape[0]
    abn = tf.reshape(abn, [n_samples, batch_size, -1])
    apn = p_fn.sample_n(s, n_samples)[1]
    return mmd(apn, abn, kernel)


##### Stein Discrepency #####

def rbf_kernel(x, dim, h=1.):
    # Reference 1: https://github.com/ChunyuanLI/SVGD/blob/master/demo_svgd.ipynb
    # Reference 2: https://github.com/yc14600/svgd/blob/master/svgd.py
    pdist = tf.reduce_sum(
      tf.square(x[None] - x[:, None]), axis=-1)
    #XY = tf.matmul(x, tf.transpose(x))
    #X2_ = tf.reshape(tf.reduce_sum(tf.square(x), axis=1), shape=[tf.shape(x)[0], 1])
    #X2 = tf.tile(X2_, [1, tf.shape(x)[0]])
    #pdist = tf.subtract(tf.add(X2, tf.transpose(X2)), 2 * XY)  # pairwise distance matrix
    kxy = tf.exp(- pdist / h ** 2 / 2.0)  # kernel matrix

    sum_kxy = tf.expand_dims(tf.reduce_sum(kxy, axis=1), 1)
    x = tf.reshape(x, [-1,dim,x.shape[1]])
    dxkxy = tf.add(-tf.einsum('ija,jka->ika', kxy, x), tf.multiply(x, sum_kxy)) / (h ** 2)  # sum_y dk(x, y)/dx
    #tf.print("dxkxy.shape: {}".format(dxkxy.shape))

    dxykxy_tr = tf.multiply((dim * (h**2) - pdist), kxy) / (h**4)  # tr( dk(x, y)/dxdy )
    #tf.print("dxy_kxy_tr.shape: {}".format(dxykxy_tr.shape))

    return kxy, dxkxy, dxykxy_tr


def imq_kernel(x, dim, beta=-.5, c=1.):
    XY = tf.matmul(x, tf.transpose(x))
    X2_ = tf.reshape(tf.reduce_sum(tf.square(x), axis=1), shape=[tf.shape(x)[0], 1])
    X2 = tf.tile(X2_, [1, tf.shape(x)[0]])
    pdist = tf.subtract(tf.add(X2, tf.transpose(X2)), 2 * XY)  # pairwise distance matrix

    kxy = (c + pdist) ** beta

    coeff = 2 * beta * (c + pdist) ** (beta-1)
    dxkxy = tf.matmul(coeff, x) - tf.multiply(x, tf.expand_dims(tf.reduce_sum(coeff, axis=1), 1))

    dxykxy_tr = tf.multiply((c + pdist) ** (beta - 2),
                            - 2 * dim * c * beta + (- 4 * beta ** 2 + (4 - 2 * dim) * beta) * pdist)

    return kxy, dxkxy, dxykxy_tr

@gin.configurable
def stein(sp, x, Kernel, dim):
    kxy, dxkxy, dxykxy_tr = Kernel(x, dim)
    # tf.print(x.shape)
    sp = tf.reshape(sp, [sp.shape[0],dim,-1])
    # tf.print(sp.shape)
    t13 = tf.multiply(tf.einsum("ija,kja->ika", sp, sp), kxy) + dxykxy_tr
    # tf.print(t13.shape)
    t2_before_tr = tf.einsum("ija,kja->ika", sp, dxkxy)
    # tf.print(t2_before_tr.shape)
    t2 = 2 * tf.trace(tf.reshape(t2_before_tr, [t2_before_tr.shape[2], t2_before_tr.shape[1], -1]))
    n = tf.cast(tf.shape(x)[0], tf.float32)
    # tf.print(t2.shape)
    ksd = (tf.reduce_sum(t13, [0,1]) + t2) / (n ** 2)

    return ksd

class Stein(Divergence):
  """Stein discrepancy."""

  def primal_estimate(
      self, s, p_fn, b_fn, n_samples,
      kernel=rbf_kernel, action_spec=None):
    abn = b_fn.sample_n(s, n_samples)[1]
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(abn)
      abn_logp = p_fn.get_log_density(
        s, utils.clip_by_eps(abn, action_spec, CLIP_EPS))
    sp = tape.gradient(abn_logp, abn)
    return stein(sp, abn, kernel, dim=abn.shape[2])

  def primal_estimate_internal(
      self, s, p_fn, abn, abn_logb, n_samples,
      kernel=rbf_kernel, action_spec=None):
    batch_size = s.shape[0]
    abn = tf.reshape(abn, [n_samples, batch_size, -1])
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(abn)
      abn_logp = p_fn.get_log_density(
        s, utils.clip_by_eps(abn, action_spec, CLIP_EPS))
    sp = tape.gradient(abn_logp, abn)
    return stein(sp, abn, kernel, dim=abn.shape[2])


CLS_DICT = dict(
    kl=KL,
    w=W,
    mmd=MMD,
    stein=Stein,
    )


def get_divergence(name):
  return CLS_DICT[name]()
