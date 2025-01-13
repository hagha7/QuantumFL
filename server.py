import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import flwr as fl
from opacus import PrivacyEngine
import pennylane as qml

n_qubits = 4
encoder_depth = 2
dev = qml.device("lightning.qubit", wires=n_qubits)

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
    """
    Defines a variational quantum circuit with feature mapping, entangling gates, 
    and parameterized rotations.

    Args:
        n_qubits (int): Number of qubits.
        params (array): Parameters for the variational circuit (shape: [layers, n_qubits, 3]).
    """
    # Feature Map
    featureMap(n_qubits)
    
    # Variational layers
    for layer_params in params:
        for idx in range(n_qubits):
            qml.RY(layer_params[idx, 0], wires=idx)
            qml.RX(layer_params[idx, 1], wires=idx)
            qml.RZ(layer_params[idx, 2], wires=idx)
        
        # Add entangling gates
        entangling(n_qubits)

# 1. Define the model architecture
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(5408, 128)
        self.fc2 = nn.Linear(128, 10)
        self.encoder_params = nn.Parameter(0.01 * torch.randn(encoder_depth, n_qubits))


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


# 2. Load Federated Data
def load_federated_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    # Split the data for simulation
    num_clients = 10
    client_datasets = random_split(dataset, [len(dataset) // num_clients] * num_clients)
    return client_datasets

# 3. Differential Privacy Optimizer with Opacus
def get_private_optimizer(model):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    privacy_engine = PrivacyEngine(
        model,
        sample_rate=0.01,  # Adjust based on batch size
        noise_multiplier=0.5,
        max_grad_norm=1.0,
    )
    privacy_engine.attach(optimizer)
    return optimizer, privacy_engine

# 4. Train the Model Federated Setup
def train_model_on_client(model, data_loader, epochs=1):
    criterion = nn.CrossEntropyLoss()
    optimizer, privacy_engine = get_private_optimizer(model)
    model.train()

    for epoch in range(epochs):
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Detach privacy engine after training
    privacy_engine.detach()
    return model.state_dict()

# 5. Flower Client for Federated Learning
class MnistClient(fl.client.NumPyClient):
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        return train_model_on_client(self.model, self.data_loader, epochs=1)

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        loss, correct = 0, 0
        with torch.no_grad():
            for data, target in self.data_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target.view_as(pred)).sum().item()
        return loss, len(self.data_loader.dataset), correct

# 6. Start Federated Training
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client_datasets = load_federated_data()

    # Create Flower clients for each client dataset
    clients = []
    for dataset in client_datasets:
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        model = ConvNet().to(device)
        clients.append(MnistClient(model, data_loader))

    # Federated Learning Strategy
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=fl.server.strategy.FedAvg(
            fraction_fit=0.1,  # Fraction of clients per round
            min_fit_clients=2,
            min_available_clients=10,
        ),
    )
