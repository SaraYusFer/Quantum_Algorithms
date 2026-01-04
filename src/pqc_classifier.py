import numpy as np
import cirq
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
import matplotlib.pyplot as plt


########### Quantum circuit setup ###########

def create_qubits(n):
    """Create a register of qubits"""
    return cirq.LineQubit.range(n)

def encoding_circuit(qubits, x):
    # you say rx, ry, rz, but the code does rz, ry, rz?
    # I prefer rx, ry, rz, but the tutorial code has rz, ry, rz too
    # also, rotating like this is equivilant to rotation along a diagonal, why use the extra gates for that?
    """Data encoding layer: apply Rx, Ry, Rz rotations based on input features x"""
    for i, q in enumerate(qubits):
        yield cirq.rz(x[i])(q)        # since we start at |0>, which is on the z-axis this rotation doesn't do anything
        yield cirq.ry(x[i])(q)
        yield cirq.rz(x[i])(q)

def variational_circuit(qubits, theta, n_layers):
    """Parameterized trainable layers (ansatz)"""
    n_qubits = len(qubits)
    for l in range(n_layers):
        for i, q in enumerate(qubits):
            # we only allow rotaion along one axis? this might limit its learning capacity
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
    #n_qubits = int(np.log2(len(state_vector)))
    for i, amp in enumerate(state_vector):
        parity = (-1) ** (bin(i).count("1"))                        # this accounts to assigning 1 or -1 based on the qbit number to each qbit,
        exp += parity * np.abs(amp) ** 2                            # I don't know if that's intentional, but it seems odd to me
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
    theta = np.random.randn(n_layers * len(qubits))
    eps = 1e-3
    loss_history = []

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
        loss_history.append(current_loss)
        print(f"Epoch {epoch+1}, Loss: {current_loss:.4f}")
    return theta, loss_history


def accuracy(X, y, theta, qubits, simulator, n_layers=3):
    preds = np.array([classifier(x, theta, qubits, simulator, n_layers) for x in X])
    return np.mean(preds == y)

########### Visualization ###########

def plot_dataset(X, y, title="Dataset", ax=None):
    """Plot 2D dataset with binary labels"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))
    reds = y == -1
    blues = y == 1
    ax.scatter(X[reds, 0], X[reds, 1], c="red", s=30, edgecolor="k", label="-1")
    ax.scatter(X[blues, 0], X[blues, 1], c="blue", s=30, edgecolor="k", label="+1")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)
    ax.legend()

def plot_decision_boundary(theta, qubits, simulator, X, y, n_layers=3, resolution=100):
    """Plot the learned decision boundary of the PQC classifier"""
    x_min, x_max = X[:,0].min() - 0.2, X[:,0].max() + 0.2
    y_min, y_max = X[:,1].min() - 0.2, X[:,1].max() + 0.2

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([classifier(pt, theta, qubits, simulator, n_layers) for pt in grid])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6,5))
    plt.contourf(xx, yy, Z, alpha=0.3, levels=[-1,0,1], colors=['red','blue'])
    plt.scatter(X[y==-1,0], X[y==-1,1], c='red', edgecolor='k', label='-1')
    plt.scatter(X[y==1,0], X[y==1,1], c='blue', edgecolor='k', label='+1')
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Decision boundary")
    plt.legend()
    plt.show()

def plot_training_loss(losses):
    """Plot training loss over epochs"""
    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training loss over epochs")
    plt.grid(True)
    plt.show()


########### Main ###########

def main():
    # Dataset
    X, y = make_moons(n_samples=200, noise=0.1)
    y = 2 * y - 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Quantum setup
    n_qubits = 2
    n_layers = 3
    qubits = create_qubits(n_qubits)
    simulator = cirq.Simulator()

    # Plot initial dataset
    plot_dataset(X_train, y_train, title="Training Dataset")

    # Training
    theta, loss_history = train(X_train, y_train, qubits, simulator, n_layers=n_layers, lr=0.1, epochs=20)

    # Plot training loss
    plot_training_loss(loss_history)

    # Evaluation
    test_acc = accuracy(X_test, y_test, theta, qubits, simulator, n_layers=n_layers)
    print("Test accuracy:", test_acc)

    # Plot decision boundary
    plot_decision_boundary(theta, qubits, simulator, X_train, y_train, n_layers=n_layers)


if __name__ == "__main__":
    main()
