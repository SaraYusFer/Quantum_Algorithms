from simulated_circuit import SimulatedCircuit
from trainable_circuit import TrainableCircuit
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from pairty_metric import ParityMetric

def make_data(target_circuit, batchsize, seed):
  tf.random.set_seed(seed)
  n_qbits = target_circuit.layer.n_qbits
  while True:
    input_values = tf.random.uniform((batchsize, n_qbits), 0, 2 * np.pi, dtype=tf.float32)
    yield input_values, target_circuit(input_values)

def recreate_circuit(n_qbits, n_layers, n_qbits_per_small, repetitions_per_small, epochs=8, batchsize=32, seed=None, scale_inputs=False):
  #It's the target so not trainable
  big_to_replicate = TrainableCircuit(n_qbits, n_layers, trainable=False, seed=seed, scale_inputs=scale_inputs)
  # the simulated circuit, has the same n_qbits for input and same depth for fair comparison
  simulator = SimulatedCircuit(n_qbits, n_layers, n_qbits_small=n_qbits_per_small, repetitions_per_small=repetitions_per_small,
                               seed=seed, scale_inputs=scale_inputs)
  # make data
  make_data_helper = lambda: make_data(big_to_replicate, batchsize, seed)
  dataset = tf.data.Dataset.from_generator(
              make_data_helper,              output_signature=(
                  tf.TensorSpec(shape=(batchsize, n_qbits), dtype=tf.float32),
                  tf.TensorSpec(shape=(batchsize, 1), dtype=tf.float32)))
  
  simulator.compile(optimizer = "adam", loss = tf.keras.losses.MeanSquaredError(), metrics=[ParityMetric])
  history = simulator.fit(dataset, epochs=epochs, steps_per_epoch=32)
  print(simulator.layer.get_weights())
  return simulator, big_to_replicate, history

def plot_performances(name='performance', n_qbits=2, qbits_per_small=1, n_layers=1, repetitions_set=[4],
                      batchsize=100, epochs=100, scale_inputs=False, seed=None):
  fig, ax1 = plt.subplots()
  fig.suptitle(f'{n_qbits}-qubit model simulated by {qbits_per_small}-qubit models')
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('MSE loss (solid)')
  ax1.set_yscale('log')
  ax_twin = ax1.twinx()
  ax_twin.set_ylabel('Accuracy (dashed)')
  ax_twin.set_ylim((0, 1.05))
  # make sure all circuits learn the same data
  seed = tf.keras.random.randint((), 0, 38746, seed=seed)
  lines = []
  for i, reps in enumerate(repetitions_set):
    # each smaller ciruit (set of inputs) gets reps different versions
    _, _, history = recreate_circuit(n_qbits, n_layers, qbits_per_small, reps, epochs, batchsize, int(seed), scale_inputs)
    c = 'C' + str(i)
    lines += ax1.plot(history.history['loss'], label=str(reps) + ' repetitions', color=c)
    ax_twin.plot(history.history['parity_metric'], color=c, linestyle='--')
  plt.legend(handles=lines, loc='lower left')
  plt.savefig(name + '.png')

if __name__ == "__main__":
  plot_performances(repetitions_set=[4], n_qbits=2, qbits_per_small=1, batchsize=50, epochs=100, seed=5562)