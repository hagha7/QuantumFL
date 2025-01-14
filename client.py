import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import flwr as fl
import pennylane as qml

# Quantum and classical model settings
n_qubits = 4
encoder_depth = 2
dev = qml.device("lightning.qubit", wires=n_qubits)

# Define quantum feature map and circuit
def featureMap(n_qubits):
    for idx in range(n_qubits):
        qml.Hadamard(wires=idx)
        qml.RY(0.1 * idx, wires=idx)

def entangling(n_qubits):
    for i in range(0, n_qubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, n_qubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])

def variationalCircuit(n_qubits, params):
    featureMap(n_qubits)
    for layer_params in params:
        for idx in range(n_qubits):
            qml.RY(layer_params[idx, 0], wires=idx)
            qml.RX(layer_params[idx, 1], wires=idx)
            qml.RZ(layer_params[idx, 2], wires=idx)
        entangling(n_qubits)
    # Add a measurement (e.g., expectation value)
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]


# Define the hybrid quantum-classical model
class QuantumConvNet(nn.Module):
    def __init__(self):
        super(QuantumConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(5408, 128)
        self.fc2 = nn.Linear(128, 10)
        self.encoder_params = nn.Parameter(0.01 * torch.randn(encoder_depth, n_qubits, 3))

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))

        # Define QNode
        def quantum_layer(params):
            return variationalCircuit(n_qubits=n_qubits, params=params)

        # Create the QNode with correct arguments
        qnode = qml.QNode(quantum_layer, dev, interface="torch")

        # Quantum output
        quantum_output = qnode(self.encoder_params)

        quantum_output = torch.tensor(quantum_output, device=x.device)

        # Combine quantum output with classical processing
        x = x + quantum_output.sum()
        x = torch.softmax(self.fc2(x), dim=1)
        return x

# Load data for the client
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    return train_loader, val_loader

# Define the Flower client
class QuantumClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_parameters(self, config=None):  # Add `config=None` to the method signature
        print("[Client] Sending model parameters...")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]


    def set_parameters(self, parameters):
        print("[Client] Receiving and setting model parameters...")
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        print("[Client] Training model...")
        self.set_parameters(parameters)
        self.train(epochs=1)
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        print("[Client] Evaluating model...")
        self.set_parameters(parameters)
        total_loss, correct, num_samples = 0.0, 0, 0
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                total_loss += criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                num_samples += data.size(0)
        avg_loss = total_loss / num_samples
        accuracy = correct / num_samples
        print(f"[Client] Evaluation results - Loss: {avg_loss}, Accuracy: {accuracy}")
        return avg_loss, num_samples, {"accuracy": accuracy}

    def train(self, epochs):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()
        for epoch in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

# Main function to start the client
if __name__ == "__main__":
    train_loader, val_loader = load_data()
    model = QuantumConvNet()
    client = QuantumClient(model, train_loader, val_loader)
    print("[Client] Starting client...")
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
