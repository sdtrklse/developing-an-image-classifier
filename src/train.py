import argparse

import torch
from torch import nn, optim
from torchvision import models

from utils import reproduce, load_data, train_loop, test_loop, save_checkpoint

print("torch.__version__:", torch.__version__)
print("torch.cuda.is_available():", torch.cuda.is_available())


def train_model(
    data_dir: str,
    arch: str = "vgg16",
    hidden_units: int = 512,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    epochs: int = 5,
    gpu: bool = False
) -> None:
    """
    Train a neural network model on a dataset.
    
    Args:
        data_dir (str): Directory containing the dataset.
        arch (str): Model architecture to use (default is "vgg16").
        hidden_units (int): Number of hidden units in the classifier (default is 512).
        batch_size (int): Batch size for training (default is 64).
        learning_rate (float): Learning rate for the optimizer (default is 1e-3).
        epochs (int): Number of epochs to train for (default is 5).
        gpu (bool): Whether to use GPU for training (default is False).
    
    Returns:
        None
    """
    
    # Reproducibility settings
    reproduce()
    
    # Set the device to be used for training
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    
    # Load data and model
    train_loader, val_loader, test_loader, class_to_idx = load_data(data_dir, batch_size=batch_size)
    model = getattr(models, arch)(weights="DEFAULT")
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Rebuild classifier
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    
    # Move model to the device
    model.to(device)
    
    # Define the loss function
    criterion = nn.NLLLoss()
    
    # Set up the optimizer
    optimizer = optim.Adam(
        model.classifier.parameters(),  # Optimize only the classifier's parameters
        lr=learning_rate
    )
    
    print("\nTraining started...\n")
    
    # Train the model using the training loop
    train_loop(train_loader, val_loader, model, criterion, optimizer, epochs, device)
    
    print("\nTesting started...\n")
    
    # Test the model using the test loop
    test_loop(test_loader, model, criterion, device)
    
    # Save the model checkpoint
    save_checkpoint(model, arch, hidden_units, optimizer, epochs, class_to_idx)


if __name__ == "__main__":
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a neural network model on a dataset")
    
    # Add arguments
    parser.add_argument("data_dir", help="Dataset directory")
    parser.add_argument("--arch", default="vgg16", help="Model architecture")
    parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the training function with parsed arguments
    train_model(
        args.data_dir,
        args.arch,
        args.hidden_units,
        args.batch_size,
        args.learning_rate,
        args.epochs,
        args.gpu
    )
