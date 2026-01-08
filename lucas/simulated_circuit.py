import tensorflow as tf
from trainable_circuit import TrainableCircuitLayer
import numpy as np

class SimulatedCircuit(tf.keras.Model):
  def __init__(self, n_qbits, n_layers, n_qbits_small, repetitions_per_small, seed=None, scale_inputs=True):
    super().__init__()
    self.layer = SimulatedCircuitLayer(n_qbits, n_layers, n_qbits_small, repetitions_per_small, seed, scale_inputs)
  
  def call(self, xs):
    return self.layer(xs)

class SimulatedCircuitLayer(tf.keras.layers.Layer):
  def __init__(self, n_qbits, n_layers, n_qbits_small, repetitions_per_small, seed=None, scale_inputs=True):
    super().__init__()
    self.n_qbits, self.n_layers, self.n_qbits_small, self.n_per_sub = n_qbits, n_layers, n_qbits_small, repetitions_per_small

    # lambdas are positive and have to sum to one as per the origiginal lemma in the paper
    # letting them be negative is fine, as the sub models can learn that same transformation anyways (rotation of pi along x or y)
    # letting them sum to one is no longer ideal, as we only use a subset, and minimize error. we might need to sum to something
    # close to but not quite 1 for the error to be minimal, but let's initialise the sum as one
    lambdas_init = tf.ones(shape=[self.n_per_sub - 1, 1], dtype=tf.float32) / np.sqrt(self.n_per_sub)
    self.lambdas = self.add_weight((self.n_per_sub - 1, 1), lambdas_init, trainable=True, name="lambdas")

    # give all circuits a random seed based on the 'main' seed
    seeds = tf.keras.random.randint((self.n_qbits // self.n_qbits_small, self.n_per_sub), 0, 100000, seed=seed)
    self.circuits = [
      # self.n_per_sub many circuits for the first portion of the input
      [TrainableCircuitLayer(self.n_qbits_small, self.n_layers, seed=seeds[j][i], scale_inputs=scale_inputs) 
                for i in range(self.n_per_sub)]
      # number of circuits we need at least, skipping remainder
      for j in range(self.n_qbits // self.n_qbits_small)
      ]
    # handle possible remainder
    self.remainder = self.n_qbits % self.n_qbits_small
    if self.remainder > 0:
      seeds = tf.keras.random.randint((self.n_per_sub,), 0, 100000, seed=int(seeds[-1][-1]))
      self.circuits.append([TrainableCircuitLayer(self.remainder, self.n_layers, seed=seeds[i], scale_inputs=scale_inputs)
                             for i in range(self.n_per_sub)])
  
  def chop_data(self, data):
    result = []
    begin = [0, 0]
    while begin[1] + self.n_qbits_small <= self.n_qbits:
      result.append(tf.slice(data, begin, [-1, self.n_qbits_small]))
      begin[1] += self.n_qbits_small
    if self.remainder > 0:
      result.append(tf.slice(data, begin, [-1, self.remainder]))
    return result

  def call(self, xs):
    chopped_data = self.chop_data(xs)
    # summing loop
    result = tf.zeros(shape=[xs.shape[0], 1], dtype=xs.dtype)
    for sub_index in range(self.n_per_sub - 1):
      # product loop
              # we project onto Z which always gives a real value, so this is okay
      product = tf.ones(shape=[xs.shape[0], 1], dtype=xs.dtype)
      for data_index, data in enumerate(chopped_data):
        output = self.circuits[data_index][sub_index](data)
        product = tf.multiply(product, output)
      result += (self.lambdas[sub_index][0] ** 2) * product
    product = tf.ones(shape=[xs.shape[0], 1], dtype=xs.dtype)
    for data_index, data in enumerate(chopped_data):
      output = self.circuits[data_index][-1](data)
      product = tf.multiply(product, output)
    result += (1 - tf.math.reduce_sum(self.lambdas ** 2)) * product
    return result