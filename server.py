import flwr as fl

# Define a custom evaluation function
def evaluate_fn(server_round, parameters_ndarrays, config):
    print(f"[Server] Manually triggering evaluation for round {server_round}...")
    # Simulated evaluation: Replace with actual evaluation logic if needed
    loss = 0.5  # Example loss
    accuracy = 0.8  # Example accuracy
    print(f"[Server] Evaluation complete - Loss: {loss}, Accuracy: {accuracy}")
    return loss, {"accuracy": accuracy}

# Define the strategy for federated learning
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.1,  # Fraction of clients for training
    min_fit_clients=2,  # Minimum number of clients for training
    min_available_clients=2,  # Minimum number of clients needed
    evaluate_fn=evaluate_fn,  # Updated evaluation function
)

# Start the server
if __name__ == "__main__":
    print("[Server] Starting server...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )
