# Quantum Federated Learning
This project combines federated learning (FL) and quantum computing to explore distributed training with hybrid quantum-classical models. Using the Flower framework for FL and PennyLane for quantum circuits, the system demonstrates federated training of quantum-enhanced convolutional neural networks (CNNs) on a distributed MNIST dataset.

Features

- Federated Learning: Distributed training across multiple clients using the Flower framework.
- Hybrid Models: Incorporates a quantum variational circuit layer within a classical CNN architecture.
- Quantum Integration: Utilizes PennyLane for quantum circuit simulation on lightning.qubit.
- Differential Privacy (Optional): Opacus library integration for adding privacy to federated updates.
- Customizable: Flexible model, dataset, and training configuration for experimentation.
