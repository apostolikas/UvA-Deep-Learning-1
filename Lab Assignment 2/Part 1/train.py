################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-14
################################################################################

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
from copy import deepcopy


from cifar100_utils import get_train_validation_set, get_test_set


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Get the pretrained ResNet18 model on ImageNet from torchvision.models

    model18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    for param in model18.parameters():
        param.requires_grad = False

    num_ftrs = model18.fc.in_features
    model18.fc = nn.Linear(num_ftrs, 100, bias=False)

    # Randomly initialize and modify the model's last layer for CIFAR100.

    nn.init.normal_(model18.fc.weight,mean=0.0, std=0.01)

    model = model18

    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
    Returns:
        model: Model that has performed best on the validation set.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Load the datasets

    train_dataset, val_dataset = get_train_validation_set(data_dir='data/', validation_size=5000, augmentation_name=augmentation_name)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=4)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,shuffle=False, num_workers=4)

    # Initialize the optimizer (Adam) to train the last layer of the model.

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)

    # Training loop with validation after each epoch. Save the best model.

    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0 
    best_model = None
    model = model.to(device)

    for epoch in range(epochs):
        print("Epoch" ,epoch+1, "starts")
        model.train()
        for inputs,labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  

    # Load the best model on val accuracy and return it.
    #pass
        val_acc = evaluate_model(model,valloader,device)
        print(f'Validation accuracy: {100 * val_acc} %')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = deepcopy(model)
    
    #######################
    # END OF YOUR CODE    #
    #######################

    return best_model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set model to evaluation mode (Remember to set it back to training mode in the training loop)

    model.eval()

    # Loop over the dataset and compute the accuracy. Return the accuracy
    # Remember to use torch.no_grad().
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum()
            total += labels.size(0)
    accuracy = correct/total

    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def main(lr, batch_size, epochs, data_dir, seed, augmentation_name):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set the seed for reproducibility
    set_seed(seed)

    # Set the device to use for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = get_model()

    # Get the augmentation to use
    augmentation_name = augmentation_name

    # Train the model
    best_model = train_model(model.to(device), lr = lr, batch_size = batch_size, epochs = epochs, data_dir = data_dir, checkpoint_name=None, device = device, augmentation_name=augmentation_name)

    # Evaluate the model on the test set
    testset = get_test_set(data_dir)
    testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size,shuffle=False, num_workers=4)

    acc = evaluate_model(best_model,testloader,device)
    print(f'Test accuracy: {100 * acc} %')

    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=123, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR100 dataset.')
    parser.add_argument('--augmentation_name', default=None, type=str,
                        help='Augmentation to use.')

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
