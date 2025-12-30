import numpy as np
import cirq
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

def create_qubits(n):
    return [cirq.GridQubit(i, 0) for i in range(n)]

def encode_data(circuit, qubits, x):
    for i, q in enumerate(qubits):
        circuit.append(cirq.rx(x[i])(q))
        circuit.append(cirq.ry(x[i])(q))

def trainable_ansatz(circuit, qubits, theta):
    idx = 0
    for q in qubits:
        circuit.append(cirq.ry(theta[idx])(q))
        idx += 1

    # entangling layer
    for i in range(len(qubits) - 1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))

def pqc_circuit(qubits, x, theta):
    circuit = cirq.Circuit()
    encode_data(circuit, qubits, x)
    trainable_ansatz(circuit, qubits, theta)
    return circuit

def expectation_z_n(state_vector, n):
    exp = 0.0
    for i, amp in enumerate(state_vector):
        parity = (-1) ** (bin(i).count("1"))
        exp += parity * np.abs(amp) ** 2
    return exp

simulator = cirq.Simulator()

def forward(x, theta, qubits):
    circuit = pqc_circuit(qubits, x, theta)
    result = simulator.simulate(circuit)
    return expectation_z_n(result.final_state_vector, len(qubits))

def loss(theta, X, y, qubits):
    preds = np.array([forward(x, theta, qubits) for x in X])
    return np.mean((preds - y) ** 2)

def train(X, y, qubits, lr=0.1, epochs=30):
    theta = np.random.randn(len(qubits))
    eps = 1e-3

    for epoch in range(epochs):
        grads = np.zeros_like(theta)
        for i in range(len(theta)):
            t_plus = theta.copy()
            t_minus = theta.copy()
            t_plus[i] += eps
            t_minus[i] -= eps

            grads[i] = (loss(t_plus, X, y, qubits) -
                        loss(t_minus, X, y, qubits)) / (2 * eps)

        theta -= lr * grads
        print(f"Epoch {epoch}, Loss: {loss(theta, X, y, qubits):.4f}")

    return theta

X, y = make_moons(n_samples=200, noise=0.1)
y = 2 * y - 1  # {0,1} → {-1,1}

X_train, X_test, y_train, y_test = train_test_split(X, y)

qubits = create_qubits(2)
theta = train(X_train, y_train, qubits)

def accuracy(X, y, theta, qubits):
    preds = np.sign([forward(x, theta, qubits) for x in X])
    return np.mean(preds == y)

print("Test accuracy:", accuracy(X_test, y_test, theta, qubits))




