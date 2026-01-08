import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np

class TrainableCircuit(tf.keras.Model):
  def __init__(self, n_qbits, n_layers, trainable=True, seed=None, scale_inputs=True):
    super().__init__()
    self.layer = TrainableCircuitLayer(n_qbits, n_layers, trainable, seed, scale_inputs)
  
  def call(self, xs):
    return self.layer(xs)

class TrainableCircuitLayer(tf.keras.layers.Layer):
  def __init__(self, n_qbits, n_layers, trainable=True, seed=None, scale_inputs=True):
    super().__init__()
    self.n_qbits, self.n_layers, self.scale_inputs = n_qbits, n_layers, scale_inputs
    self.qbits = cirq.GridQubit.rect(1, self.n_qbits)
    self.circuit = cirq.Circuit()
              # parameters
    self.theta_symbols = [[sympy.symbols(f'theta({layer}_{qbit_nr}_0:3)') for qbit_nr in range(self.n_qbits)] for layer in range(self.n_layers)]
    self.theta_symbols = np.asarray(self.theta_symbols).reshape((self.n_layers, self.n_qbits, 3))

    self.input_symbols = np.array(sympy.symbols(f'x(0:{self.n_qbits})'))

    # as per the exercise.
    self.circuit += self.encoding_layer(self.qbits, self.input_symbols)
    for symbols in self.theta_symbols:
      self.circuit += self.entangle_layer(self.qbits)
      self.circuit += self.trainable_layer(self.qbits, symbols)

    measurement = [cirq.Z(self.qbits[0])]
    for qbit in self.qbits[1:]:
      measurement[0] *= cirq.Z(qbit)
    self.outputs = tfq.layers.ControlledPQC(self.circuit, measurement)
    # find to what indexes parameters should map
    all_params = self.theta_symbols.flatten()
    all_params = np.append(all_params, self.input_symbols)
    self.param_order = np.array([self.outputs.symbols.index(param) for param in all_params])

    # innit random theta values
    innit_theta = tf.random_uniform_initializer(0, 2 * np.pi, seed=seed)
    self.thetas = self.add_weight((1, np.prod(self.theta_symbols.shape)), innit_theta, dtype=tf.float32,
        trainable=trainable, name="thetas")
    
    if self.scale_inputs:
      innit_alphas = tf.ones((1, self.n_qbits), dtype=tf.float32)
      self.alphas = self.add_weight((1, self.n_qbits), innit_alphas, dtype=tf.float32, trainable=trainable, name="alphas")
  
  # can learn any unitary operation
  def one_qbit_rotation(self, qbit, symbols):
    assert len(symbols) == 3
    return [cirq.rx(symbols[0])(qbit), cirq.ry(symbols[1])(qbit), cirq.rz(symbols[2])(qbit)]
  
  def trainable_layer(self, qbits, symbols):
    assert len(qbits) == symbols.shape[0] and symbols.shape[1] == 3
    layer = []
    for qbit, symbol_set in zip(qbits, symbols):
      layer += self.one_qbit_rotation(qbit, symbol_set)
    return layer

  def entangle_layer(self, qbits):
    assert len(qbits) > 0
    if len(qbits) == 1:
      return []
    n = len(qbits)
    if n == 2:
      return [cirq.CZ(qbits[0], qbits[1])]
    return [cirq.CZ(qbits[i], qbits[(i + 1) % n]) for i in range(n)]
  
  def encoding_layer(self, qbits, symbols):
    assert len(qbits) == len(symbols)
    # rx and ry are equally fine, we can't use rz as it won't move anything
    return [cirq.ry(symbol)(qbit) for qbit, symbol in zip(qbits, symbols)]

  def call(self, xs):
    repeated_thetas = tf.tile(self.thetas, [xs.shape[0], 1])
    # scale the inputs if needed
    if self.scale_inputs:
      repeated_alphas = tf.tile(self.alphas, [xs.shape[0], 1])
      xs = tf.multiply(xs, repeated_alphas)
    full_input = tf.concat([repeated_thetas, xs], axis=1)
    # shuffle according to the order the ControlledPQC uses
    full_input = tf.gather(full_input, self.param_order, axis=1)
    empty_data = tfq.convert_to_tensor([cirq.Circuit()] * xs.shape[0])
    res = self.outputs([empty_data, full_input])
    return res