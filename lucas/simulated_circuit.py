import tensorflow as tf
from controlled_trainableLayer import ControledTrainable
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
    # lambdas are complex weights of each tensor multiplication between the different learned unitaries
    lambdas_init = tf.ones(shape=[self.n_per_sub, 1], dtype=tf.float32) / np.sqrt(self.n_per_sub)
    self.lambdas_real = self.add_weight((self.n_per_sub, 1), lambdas_init, trainable=True, name="lambdas_real")
    lambdas_init = tf.zeros(shape=[self.n_per_sub, 1], dtype=tf.float32)
    self.lambdas_imag = self.add_weight((self.n_per_sub, 1), lambdas_init, trainable=True, name="lambdas_imag")

    # give all circuits a random seed based on the 'main' seed
    seeds = tf.keras.random.randint((self.n_qbits // self.n_qbits_small, self.n_per_sub), 0, 100000, seed=seed)
    self.circuits = [
      # self.n_per_sub many circuits for the first portion of the input
      [ControledTrainable(self.n_qbits_small, self.n_layers, seed=seeds[j][i], scale_inputs=scale_inputs) 
                for i in range(self.n_per_sub)]
      # number of circuits we need at least, skipping remainder
      for j in range(self.n_qbits // self.n_qbits_small)
      ]
    # handle possible remaining qubits
    self.remainder = self.n_qbits % self.n_qbits_small
    if self.remainder > 0:
      seeds = tf.keras.random.randint((self.n_per_sub,), 0, 100000, seed=int(seeds[-1][-1]))
      self.circuits.append([ControledTrainable(self.remainder, self.n_layers, seed=seeds[i], scale_inputs=scale_inputs)
                             for i in range(self.n_per_sub)])
  
  # a function that slices the input into smaller pieces according to the chosen number of smaller models
  def chop_data(self, data):
    result = []
    begin = [0, 0]
    while begin[1] + self.n_qbits_small <= self.n_qbits:
      result.append(tf.slice(data, begin, [-1, self.n_qbits_small]))
      begin[1] += self.n_qbits_small
    if self.remainder > 0:
      result.append(tf.slice(data, begin, [-1, self.remainder]))
    return result

  # forward pass of the layer, it combines the smaller circuit outputs into a single big output
  def call(self, xs):
    chopped_data = self.chop_data(xs)
    # summing loop
    result = tf.zeros(shape=[xs.shape[0], 1], dtype=tf.complex64)
    for sub_index in range(self.n_per_sub):
      # product loop
      product = tf.ones(shape=[xs.shape[0], 1], dtype=tf.complex64)
      for data_index, data in enumerate(chopped_data):
        output = self.circuits[data_index][sub_index](data)
        product = tf.multiply(product, output)
      result += tf.complex(self.lambdas_real[sub_index][0], self.lambdas_imag[sub_index][0]) * product
    return tf.math.real(result)