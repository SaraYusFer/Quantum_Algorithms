import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np

class ControledTrainable(tf.keras.Model):
  def __init__(self, n_qbits, n_layers, trainable=True, seed=None, scale_inputs=True):
    super().__init__()
    self.layer = ControledTrainableLayer(n_qbits, n_layers, trainable, seed, scale_inputs)
  
  def call(self, xs):
    return self.layer(xs)

class ControledTrainableLayer(tf.keras.layers.Layer):
  '''
    Trainable layer that can give the real and imaginary part of its expected value
  '''
  def __init__(self, n_qbits, n_layers, trainable=True, seed=None, scale_inputs=True):
    super().__init__()
    self.n_qbits, self.n_layers, self.scale_inputs = n_qbits, n_layers, scale_inputs
    # we need a circuit for both the real and imaginary part of the projection
    self.RIqbits = [cirq.GridQubit.rect(1, self.n_qbits + 1), cirq.GridQubit.rect(1, self.n_qbits + 1)]
    self.RICircuit = [cirq.Circuit(), cirq.Circuit()]
              # parameters
    self.theta_symbols = [[sympy.symbols(f'theta({layer}_{qbit_nr}_0:3)') for qbit_nr in range(self.n_qbits)] for layer in range(self.n_layers)]
    self.theta_symbols = np.asarray(self.theta_symbols).reshape((self.n_layers, self.n_qbits, 3))
    # zeta is the value that changes on circuit into its dagger (and possible into another unitary altogether)
    # this is in line with the paper, but very much bullshit in my opinion. It is a very limited approach
    self.zeta_symbols = np.asarray(sympy.symbols(f'zeta(0:2)'))

    # see trainable_circuit.py for descriptions on the rest
    self.input_symbols = np.asarray(sympy.symbols(f'x(0:{self.n_qbits})'))

    for i, circuit in enumerate(self.RICircuit):
      circuit += self.encoding_layer(self.RIqbits[i][:-1], self.input_symbols)
      # add the hadamard test in to compute the real and imaginary values of the projection
      circuit += self.hadamard_test(self.RIqbits[i], self.zeta_symbols, real=(i == 0))
      for symbols in self.theta_symbols:
        circuit += self.entangle_layer(self.RIqbits[i][:-1])
        circuit += self.trainable_layer(self.RIqbits[i][:-1], symbols)

    # real and imaginary outputs
    self.outputs = [tfq.layers.ControlledPQC(self.RICircuit[0], cirq.Z(self.RIqbits[0][-1])),
                    tfq.layers.ControlledPQC(self.RICircuit[1], cirq.Z(self.RIqbits[1][-1]))]
    # find to what indexes parameters should map
    all_params = self.theta_symbols.flatten()
    all_params = np.append(all_params, self.input_symbols)
    all_params = np.append(all_params, self.zeta_symbols)
    self.param_order = np.array([self.outputs[0].symbols.index(param) for param in all_params])

    # innit random theta and zeta values
    inniter = tf.random_uniform_initializer(0, 2 * np.pi, seed=seed)
    self.thetas = self.add_weight((1, np.prod(self.theta_symbols.shape)), inniter, dtype=tf.float32,
        trainable=trainable, name="thetas")
    self.zetas = self.add_weight((1, 2), inniter, dtype=tf.float32, trainable=trainable, name="zetas")
    
    if self.scale_inputs:
      innit_alphas = tf.ones((1, self.n_qbits), dtype=tf.float32)
      self.alphas = self.add_weight((1, self.n_qbits), innit_alphas, dtype=tf.float32, trainable=trainable, name="alphas")
  
  # see wikipedia, it is explained well there
  def hadamard_test(self, qbits, zetas, real):
    circuit = cirq.Circuit(cirq.H(qbits[-1]))
    if not real:
      circuit += cirq.S(qbits[-1])
    circuit += cirq.rz(zetas[0])(qbits[0]).controlled_by(qbits[-1])
    circuit += cirq.X(qbits[-1])
    circuit += cirq.rz(zetas[1])(qbits[0]).controlled_by(qbits[-1])
    circuit += cirq.X(qbits[-1])
    circuit += cirq.H(qbits[-1])
    return circuit

  # see trainable_circuit.py
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
    repeated_zetas = tf.tile(self.zetas, [xs.shape[0], 1])
    # scale the inputs if needed
    if self.scale_inputs:
      repeated_alphas = tf.tile(self.alphas, [xs.shape[0], 1])
      xs = tf.multiply(xs, repeated_alphas)
    full_input = tf.concat([repeated_thetas, xs, repeated_zetas], axis=1)
    # shuffle according to the order the ControlledPQC uses
    full_input = tf.gather(full_input, self.param_order, axis=1)
    empty_data = tfq.convert_to_tensor([cirq.Circuit()] * xs.shape[0])
    real = self.outputs[0]([empty_data, full_input])
    imag = self.outputs[1]([empty_data, full_input])
    return tf.complex(real, imag)