import os
import random
from typing import Dict, NamedTuple

import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

os.environ["KERAS_BACKEND"] = "torch"
import keras


class DataLoaders(NamedTuple):
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_to_idx: Dict[str, int]


def load_data(data_dir: str, batch_size: int = 64) -> DataLoaders:
    """
    Loads the dataset and returns DataLoader objects for training, validation, and testing.
    
    Args:
        data_dir (str): Path to the dataset directory, expected to contain "train/", "valid/", and "test/" subdirectories.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 64.
    
    Returns:
        DataLoaders: NamedTuple containing train, validation, test loaders, and class-to-index mapping.
    """

    # Define transformations
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        
        "valid": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    # Load datasets
    image_datasets = {
        key: datasets.ImageFolder(os.path.join(data_dir, key), transform=data_transforms[key])
        for key in ["train", "valid", "test"]
    }
    
    # Create DataLoaders
    dataloaders = {
        key: DataLoader(image_datasets[key], batch_size=batch_size, shuffle=(key == "train"))
        for key in image_datasets
    }
    
    return DataLoaders(
        train_loader=dataloaders["train"],
        val_loader=dataloaders["valid"],
        test_loader=dataloaders["test"],
        class_to_idx=image_datasets["train"].class_to_idx
    )


def train_loop(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int = 5,
    device: torch.device = torch.device("cpu")
) -> None:
    """
    Train a PyTorch model with a training loop.
    
    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        model (torch.nn.Module): The PyTorch model to train.
        criterion (torch.nn.Module): The loss function to use.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        epochs (int, optional): The number of epochs to train for (default is 5).
        device (torch.device, optional): The device to train on (default is torch.device("cpu")).
    
    Returns:
        None
    """

    for epoch in range(epochs):

        train_loss = 0.0
        train_total, train_correct = 0, 0
        
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Initialize progress bar
        n_batches = len(train_loader)
        pbar = keras.utils.Progbar(target=n_batches)
        
        # Training phase
        model.train()
        for batch, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass: Compute prediction and loss
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item() * images.size(0)
            
            # Calculate accuracy
            _, preds = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()
            train_accuracy = 100 * (train_correct / train_total)
            
            # Backward pass: Update model weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update the progress bar with each batch
            pbar.update(batch, values=[("loss", loss.item()), ("acc", train_accuracy)])
        
        # Validation phase
        val_loss = 0.0
        val_correct, val_total = 0, 0
        
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass: Compute prediction and loss
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                # Calculate accuracy
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()
            
            # Calculate and store average validation loss and accuracy
            val_loss /= val_total
            val_accuracy = 100 * (val_correct / val_total)
            
            # Final update for the progress bar with validation data
            pbar.update(n_batches, values=[("val_loss", val_loss), ("val_acc", val_accuracy)])


def test_loop(
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    device: torch.device = torch.device("cpu")
) -> None:
    """
    Evaluate a model on a test dataset with a test loop.
    
    Args:
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        model (torch.nn.Module): The PyTorch model to evaluate.
        criterion (torch.nn.Module): The loss function to use.
        device (torch.device, optional): The device to evaluate on (default is torch.device("cpu")).
    
    Returns:
        None
    """

    # Initialize variables to keep track of the total loss and accuracy
    test_loss = 0.0
    test_correct, test_total = 0, 0
    
    # Create a progress bar to track the evaluation progress
    n_batches = len(test_loader)
    pbar = keras.utils.Progbar(target=n_batches)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Iterate over the test data loader
    with torch.no_grad():
        for batch, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            
            # Get the predictions
            _, preds = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (preds == labels).sum().item()
            
            # Calculate the accuracy
            test_accuracy = 100 * (test_correct / test_total)
            
            # Update the progress bar
            pbar.update(batch, values=[("loss", loss.item()), ("acc", test_accuracy)])
            
            # Finish the progress bar with a final update
            if batch + 1 == n_batches:
                pbar.update(n_batches, values=None)
    
    print(f"Average loss: {test_loss / test_total:.4f}")


def save_checkpoint(
    model: torch.nn.Module,
    arch: str,
    hidden_units: int,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    class_to_idx: Dict[str, int],
    filepath: str = "checkpoint.pth"
) -> None:
    """
    Save a checkpoint of a PyTorch neural network model.
    
    Args:
        model (torch.nn.Module): The neural network model to save.
        arch (str): The model architecture.
        hidden_units (int): The number of hidden units in the classifier.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epochs (int): The number of epochs the model has been trained for.
        class_to_idx (Dict[str, int]): A dictionary mapping class names to indices.
        filepath (str, optional): File path for saving the checkpoint. Defaults to "checkpoint.pth".
    
    Returns:
        None
    """

    # Move model to the CPU to ensure compatibility during loading
    model.to("cpu")
    
    # Create a dictionary to store important components of the model's state
    checkpoint = {
        "state_dict": model.state_dict(),  # Model's state dictionary
        "arch": arch,  # Model architecture
        "hidden_units": hidden_units,  # Number of hidden units in the classifier
        "optimizer_state": optimizer.state_dict(),  # Optimizer's state dictionary
        "epochs": epochs,  # Number of training epochs
        "class_to_idx": class_to_idx,  # Class-to-index mapping
    }
    
    # Save the checkpoint dictionary to the specified file path
    torch.save(checkpoint, filepath)
    print(f"\nCheckpoint saved to {filepath}\n")


def load_checkpoint(filepath: str) -> torch.nn.Module:
    """
    Load a model checkpoint from a file.
    
    Args:
        filepath (str): The path to the checkpoint file.
    
    Returns:
        torch.nn.Module: The loaded model.
    """

    # Load the checkpoint dictionary from the specified file path
    checkpoint = torch.load(filepath, map_location=torch.device("cpu"), weights_only=True)
    
    # Dynamically load the correct model architecture
    model = getattr(models, checkpoint["arch"])(weights="DEFAULT")
    
    # Rebuild classifier
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, checkpoint["hidden_units"]),
        nn.ReLU(),
        nn.Linear(checkpoint["hidden_units"], 102),
        nn.LogSoftmax(dim=1)
    )
    
    # Load model state and class mapping
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]
    
    print(f"\nCheckpoint loaded from {filepath}\n")
    
    return model


def process_image(image_path: str) -> torch.Tensor:
    """
    Process an image by resizing and cropping it to the specified size,
    then normalizing the pixel values.
    
    Args:
        image_path (str): The path to the image file.
    
    Returns:
        torch.Tensor: The processed image as a tensor.
    """

    image = Image.open(image_path)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    tensor_image = transform(image)
    if not isinstance(tensor_image, torch.Tensor):
        tensor_image = transforms.ToTensor()(tensor_image)
    
    return tensor_image.unsqueeze(0)


def reproduce(seed: int = 42) -> None:
    """
    Reproduce the random seeds and configurations for PyTorch and NumPy.
    
    Args:
        seed (int, optional): The random seed to use. Defaults to 42.
    
    Returns:
        None
    """
    
    # Setting random seeds
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # NumPy random generator
    torch.manual_seed(seed)  # PyTorch random seed for CPU
    torch.cuda.manual_seed_all(seed)  # PyTorch random seed for all GPUs
    
    # The configuration ensures that PyTorch operations are deterministic on GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
