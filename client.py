import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import flwr as fl
from opacus import PrivacyEngine
import pennylane as qml

n_qubits = 4
encoder_depth = 2
dev = qml.device("lightning.qubit", wires=n_qubits)

# Define quantum circuit components
def featureMap(n_qubits):
    for idx in range(n_qubits):
        qml.Hadamard(wires=idx)
    for idx, element in enumerate(n_qubits):
        qml.RY(element, wires=idx)

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

# Define the ConvNet with Quantum Layer
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
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

        # Quantum layer
        def quantum_layer(params):
            return variationalCircuit(n_qubits, params)

        qnode = qml.QNode(quantum_layer, dev, interface="torch")
        quantum_output = qnode(self.encoder_params)

        x = x + quantum_output.sum()  # Incorporating quantum output
        x = torch.softmax(self.fc2(x), dim=1)
        return x

# Load data for this client
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


class MnistClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

    def get_parameters(self, config=None):  # Add config=None
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train(epochs=1)
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        # Set model parameters from server
        self.set_parameters(parameters)
        
        # Evaluation logic
        criterion = nn.CrossEntropyLoss()
        total_loss = 0
        correct = 0
        self.model.eval()
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        # Calculate metrics
        accuracy = correct / len(self.val_loader.dataset)
        avg_loss = total_loss / len(self.val_loader)
        
        # Return results
        return avg_loss, len(self.val_loader.dataset), {"accuracy": accuracy}


    def train(self, epochs):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        privacy_engine = PrivacyEngine(
            self.model,
            sample_rate=0.01,  # Adjust based on batch size
            noise_multiplier=0.5,
            max_grad_norm=1.0,
        )
        privacy_engine.attach(optimizer)

        self.model.train()
        for epoch in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        privacy_engine.detach()

    def test(self):
        criterion = nn.CrossEntropyLoss()
        correct = 0
        total_loss = 0
        self.model.eval()
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct / len(self.val_loader.dataset)
        return total_loss / len(self.val_loader), accuracy


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader = load_data()

    # Initialize model
    model = ConvNet().to(device)

    # Create a NumPyClient
    client = MnistClient(model, train_loader, val_loader)

    # Convert the NumPyClient to a Client object and start the Flower client
    fl.client.start_client(
        server_address="localhost:8080",
        client=client.to_client()  # Use .to_client() to convert
    )

