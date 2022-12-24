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
# Date Created: 2022-11-01
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

import torch
import matplotlib.pyplot as plt 



def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes]
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    num_classes = predictions.shape[1]
    conf_mat = np.zeros((num_classes,num_classes))
    y_pred = np.zeros((targets.shape[0]))

    for i in range(predictions.shape[0]):
      y_pred[i] = int(round(np.argmax(predictions[i,:])))
      conf_mat[int(y_pred[i]),targets[i]] += 1
    
    #######################
    # END OF YOUR CODE    #
    #######################
    return conf_mat


def confusion_matrix_to_metrics(confusion_matrix, beta=1.):
    """
    Converts a confusion matrix to accuracy, precision, recall and f1 scores.
    Args:
        confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
    Returns: a dictionary with the following keys:
        accuracy: scalar float, the accuracy of the confusion matrix
        precision: 1D float array of size [n_classes], the precision for each class
        recall: 1D float array of size [n_classes], the recall for each clas
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    
    metrics = {'accuracy':None, 'precision':None, 'recall':None, 'f1_beta':None}

    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    precision = []
    recall = []
    f1_beta01 = []
    f1_beta = []
    f1_beta10 = []

    for i in range(confusion_matrix.shape[0]):
      tp = confusion_matrix[i,i]
      fn = np.sum(confusion_matrix[:,i]) - tp
      fp = np.sum(confusion_matrix[i,:]) - tp
      tn = np.sum(confusion_matrix) - tp - fn - fp
      precision.append(tp/(tp+fp))
      recall.append(tp/(tp+fn))
      f1_beta01.append(((1+0.1**2) * (tp/(tp+fp)) * (tp/(tp+fn))) / (0.1**2 * (tp/(tp+fp)) + (tp/(tp+fn))))
      f1_beta.append(((1+beta**2) * (tp/(tp+fp)) * (tp/(tp+fn))) / (beta**2 * (tp/(tp+fp)) + (tp/(tp+fn))))
      f1_beta10.append(((1+10**2) * (tp/(tp+fp)) * (tp/(tp+fn))) / (10**2 * (tp/(tp+fp)) + (tp/(tp+fn))))


    metrics['accuracy'] = accuracy
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_beta'] = f1_beta
    metrics['f1_beta-0.1'] = f1_beta01
    metrics['f1_beta-10'] = f1_beta10
    
    
    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def evaluate_model(model, data_loader, num_classes=10):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    loss_module = CrossEntropyModule()
    evaluation_loss = 0

    cm = np.zeros((num_classes,num_classes))
    
    for inputs,labels in data_loader:
        outputs = model.forward(inputs)
        loss = loss_module.forward(outputs,labels)
        evaluation_loss += loss.item() * labels.shape[0]
        cm = cm + confusion_matrix(outputs, labels)

    metrics = confusion_matrix_to_metrics(cm, beta=1.)
    metrics['Confusion matrix'] = cm
    evaluation_loss = evaluation_loss/len(data_loader.dataset)
    metrics['eval_loss'] = evaluation_loss
    
    #######################
    # END OF YOUR CODE    #
    #######################

    return metrics



def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # TODO: Initialize model and loss module
    model = MLP(3*32*32,hidden_dims,10)
    loss_module = CrossEntropyModule()
    # TODO: Training loop including validation
    val_accuracies = []
    # TODO: Test best model
    test_accuracy = 0
    # TODO: Add any information you might want to save for plotting
    logging_info = {'Train loss':None , 'Train accuracy': None , "Valid accuracy": None}
    train_loader = cifar10_loader['train']
    val_loader = cifar10_loader['validation']
    test_loader = cifar10_loader['test']
    
    train_loss = []
    valid_loss = []
    train_accuracy = []
    best_valid_acc = 0 
    best_model = None
    
    for epoch in range(epochs):
        epoch_loss = 0
        conf_mat = np.zeros((10,10))

        for inputs,labels in train_loader:
            outputs = model.forward(inputs)
            conf_mat = conf_mat + confusion_matrix(outputs,labels)
            loss = loss_module.forward(outputs, labels)
            epoch_loss += loss.item() * labels.shape[0]
            gradient_loss = loss_module.backward(outputs,labels)
            model.backward(gradient_loss)
            for layer in model.fc_layers:
                layer.params['weight'] -=  lr * layer.grads['weight']
                layer.params['bias']   -=  lr * layer.grads['bias']

        train_metrics = confusion_matrix_to_metrics(conf_mat, beta=1.)
        train_loss.append(epoch_loss/len(train_loader.dataset))
        train_accuracy.append(train_metrics['accuracy'])
        valid_metrics = evaluate_model(model,val_loader)
        valid_loss.append(valid_metrics['eval_loss'])
        val_accuracies.append(valid_metrics['accuracy'])
      

        if valid_metrics['accuracy'] > best_valid_acc:
            best_valid_acc = valid_metrics['accuracy']
            best_model = deepcopy(model)

    logging_info['Train accuracy'] = train_accuracy
    logging_info['Valid accuracy'] = val_accuracies
    logging_info['Train loss'] = train_loss
    logging_info['Valid loss'] = valid_loss

    print(logging_info)
    
    temp_metrics = evaluate_model(best_model,val_loader)
    print(temp_metrics['accuracy'])

    test_metrics = evaluate_model(best_model, test_loader)
    print("Test accuracy : %.4f " % test_metrics['accuracy'])
    
    logging_info['Confusion matrix'] = test_metrics['Confusion matrix']
    logging_info['Precision'] = test_metrics['precision']
    logging_info['Recall'] = test_metrics['recall']
    logging_info['F1-Beta'] = test_metrics['f1_beta']
    logging_info['F1-Beta-0.1'] = test_metrics['f1_beta-0.1']
    logging_info['F1-Beta-10'] = test_metrics['f1_beta-10']

    logging_dict = logging_info

    [print(key,':',value) for key, value in logging_dict.items()]

    #######################
    # END OF YOUR CODE    #
    #######################

    return best_model, val_accuracies, test_accuracy, logging_dict


def my_plot(logging_info):
      train_loss = logging_info['Train loss']
      valid_loss = logging_info['Valid loss']

      plt.plot(train_loss)
      plt.plot(valid_loss)
      plt.ylabel('Accuracy')
      plt.xlabel('Epochs')
      plt.xticks(np.arange(1, 11, 1))
      plt.ylabel('Loss')
      plt.xlabel('Epochs')
      plt.legend(['Train', 'Validation'], loc='upper right', prop={'size': 12})
      #plt.savefig('temp1.png', dpi=fig.dpi)

      
      plt.show()

      train_accuracy = logging_info['Train accuracy']
      valid_accuracy = logging_info['Valid accuracy']
      plt.plot(train_accuracy)
      plt.plot(valid_accuracy)
      plt.ylabel('Accuracy')
      plt.xlabel('Epochs')
      plt.xticks(np.arange(1, 11, 1))
      plt.legend(['Train', 'Validation'], loc='lower right', prop={'size': 12})
      #plt.savefig('temp2.png', dpi=fig.dpi)


      plt.show()
        
      conf_matrix = logging_info['Confusion matrix']
      fig, ax = plt.subplots(figsize=(7.5, 7.5))
      ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
      for i in range(conf_matrix.shape[0]):
          for j in range(conf_matrix.shape[1]):
              ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='large')
      
      plt.xlabel('Predictions', fontsize=18)
      plt.ylabel('Actuals', fontsize=18)
      plt.title('Confusion Matrix', fontsize=18)
      #plt.savefig('temp3.png', dpi=fig.dpi)
      plt.show()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    best_model, val_accuracies, test_accuracy, logging_info = train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here

    my_plot(logging_info)  