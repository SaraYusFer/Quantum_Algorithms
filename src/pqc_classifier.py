import numpy as np
import cirq
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize


########### Quantum circuit setup ###########

def create_qubits(n):
    """Create n qubits in a 1D line"""
    return cirq.LineQubit.range(n)

def encoding_circuit(qubits, x):
    """Data encoding layer: apply Rx, Ry, Rz rotations based on input features x"""
    for i, q in enumerate(qubits):
        yield cirq.rz(x[i])(q)
        yield cirq.ry(x[i])(q)
        yield cirq.rz(x[i])(q)

def variational_circuit(qubits, theta, n_layers):
    """Parameterized trainable layers (ansatz)"""
    n_qubits = len(qubits)
    for l in range(n_layers):
        for i, q in enumerate(qubits):
            yield cirq.ry(theta[l * n_qubits + i])(q)
        for i in range(n_qubits - 1):
            yield cirq.CZ(qubits[i], qubits[i + 1])

def pqc_circuit(qubits, x, theta, n_layers=3):
    """Full PQC: U(x, theta) = U_train(theta) * U_enc(x)"""
    return cirq.Circuit(encoding_circuit(qubits, x),
                        variational_circuit(qubits, theta, n_layers))


########### Forward pass / output ###########

def expectation_z_n(state_vector):
    """Compute ⟨Z⊗n⟩ for state vector"""
    exp = 0.0
    n_qubits = int(np.log2(len(state_vector)))
    for i, amp in enumerate(state_vector):
        parity = (-1) ** (bin(i).count("1"))
        exp += parity * np.abs(amp) ** 2
    return exp

def forward(x, theta, qubits, simulator, n_layers=3):
    circuit = pqc_circuit(qubits, x, theta, n_layers)
    result = simulator.simulate(circuit)
    return expectation_z_n(result.final_state_vector)

def classifier(x, theta, qubits, simulator, n_layers=3):
    """Binary classifier h_theta(x) = sign(f_theta(x))"""
    return 1 if forward(x, theta, qubits, simulator, n_layers) >= 0 else -1


########### Loss and training ###########

def mse_loss(theta, X, y, qubits, simulator, n_layers=3):
    preds = np.array([forward(x, theta, qubits, simulator, n_layers) for x in X])
    return np.mean((preds - y) ** 2)

def train(X, y, qubits, simulator, n_layers=3, lr=0.1, epochs=30):
    """Simple gradient-descent training using finite differences."""
    theta = np.random.randn(n_layers * len(qubits))
    eps = 1e-3

    for epoch in range(epochs):
        grads = np.zeros_like(theta)
        for i in range(len(theta)):
            t_plus, t_minus = theta.copy(), theta.copy()
            t_plus[i] += eps
            t_minus[i] -= eps
            grads[i] = (mse_loss(t_plus, X, y, qubits, simulator, n_layers) -
                        mse_loss(t_minus, X, y, qubits, simulator, n_layers)) / (2 * eps)
        theta -= lr * grads
        current_loss = mse_loss(theta, X, y, qubits, simulator, n_layers)
        print(f"Epoch {epoch+1}, Loss: {current_loss:.4f}")
    return theta

def accuracy(X, y, theta, qubits, simulator, n_layers=3):
    preds = np.array([classifier(x, theta, qubits, simulator, n_layers) for x in X])
    return np.mean(preds == y)


def main():
    # Dataset
    X, y = make_moons(n_samples=200, noise=0.1)
    y = 2 * y - 1  # {0,1} -> {-1,1}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Quantum setup
    n_qubits = 2
    n_layers = 3
    qubits = create_qubits(n_qubits)
    simulator = cirq.Simulator()

    # Training
    theta = train(X_train, y_train, qubits, simulator, n_layers=n_layers, lr=0.1, epochs=20)

    # Evaluation
    test_acc = accuracy(X_test, y_test, theta, qubits, simulator, n_layers=n_layers)
    print("Test accuracy:", test_acc)

if __name__ == "__main__":
    main()
