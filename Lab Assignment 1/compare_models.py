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
This file implements the execution of different hyperparameter configurations with
respect to using batch norm or not, and plots the results
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import train_mlp_pytorch

import torch
import torch.nn as nn
import torch.optim as optim

import json
from train_mlp_pytorch import train as train_pytorch
import matplotlib.pyplot as plt 

# Hint: you might want to import some plotting libraries or similar
# You are also allowed to use libraries here which are not in the provided environment.


def train_models(results_filename):
    """
    Executes all requested hyperparameter configurations and stores all results in a file.
    Note that we split the running of the model and the plotting, since you might want to 
    try out different plotting configurations without re-running your models every time.

    Args:
      results_filename - string which specifies the name of the file to which the results
                         should be saved.

    TODO:
    - Loop over all requested hyperparameter configurations and train the models accordingly.
    - Store the results in a file. The form of the file is left up to you (numpy, json, pickle, etc.)
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # TODO: Run all hyperparameter configurations as requested
    results = None
    # TODO: Save all results in a file with the name 'results_filename'. This can e.g. by a json file
    
    hidden_dim = [[128], [256, 128], [512, 256, 128]]
    use_batch_norm = False
    lr = 0.1
    batch_size = 128
    epochs = 20
    results_list = []

    for hid_dim in hidden_dim:        
        _ , _, _, logging_info = train_pytorch(hidden_dims = hid_dim, lr = lr, use_batch_norm = use_batch_norm, batch_size = batch_size , epochs = epochs, seed = 42, data_dir='data/')
        #[print(key,':',value) for key, value in logging_info.items()]
        
        results_list.append(str(hid_dim))
        results = logging_info            
        results_list.append(results)
        json_file = results_list

    with open(results_filename, 'w') as f:
        json.dump(json_file, f, indent = 2)

    #######################
    # END OF YOUR CODE    #
    #######################


def plot_results(results_filename):
    """
    Plots the results that were exported into the given file.

    Args:
      results_filename - string which specifies the name of the file from which the results
                         are loaded.

    TODO:
    - Visualize the results in plots

    Hint: you are allowed to add additional input arguments if needed.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    #pass

    with open(results_filename, 'r') as f:
        results = json.load(f)

    train_loss128 = results[1]['Train loss']
    train_loss128256 = results[3]['Train loss']
    train_loss128256512 = results[5]['Train loss']

    train_acc128 = results[1]['Train accuracy']
    train_acc128256 = results[3]['Train accuracy']
    train_acc128256512 = results[5]['Train accuracy']

    valid_loss128 = results[1]['Valid loss']
    valid_loss128256 = results[3]['Valid loss']
    valid_loss128256512 = results[5]['Valid loss']

    valid_acc128true = results[1]['Valid accuracy']
    valid_acc128256true = results[3]['Valid accuracy']
    valid_acc128256512true = results[5]['Valid accuracy']

    fig1 = plt.figure(layout='constrained')
    fig1.set_figheight(5)
    fig1.set_figwidth(20)

    fig1.add_subplot(1, 3, 1)
    plt.plot(train_loss128)
    plt.plot(train_loss128256)
    plt.plot(train_loss128256512)
    plt.title('Train loss for different hidden dims')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.xticks(np.arange(1, 21, 1))
    plt.legend(['128', '256-128', '512-256-128'], loc='upper center', prop={'size': 8})


    fig1.add_subplot(1, 3, 3)
    plt.plot(valid_loss128)
    plt.plot(valid_loss128256)
    plt.plot(valid_loss128256512)
    plt.title('Valid loss for different hidden dims')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.xticks(np.arange(1, 21, 1))
    plt.legend(['128', '256-128', '512-256-128'], loc='upper center', prop={'size': 8})

    #plt.savefig('temp15.png', dpi=fig.dpi)

    plt.show()

    fig2 = plt.figure(layout='constrained')
    fig2.set_figheight(5)
    fig2.set_figwidth(20)

    fig2.add_subplot(1, 3, 1)
    plt.plot(train_acc128)
    plt.plot(train_acc128256)
    plt.plot(train_acc128256512)
    plt.title('Train acc for different hidden dims')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.xticks(np.arange(1, 21, 1))
    plt.legend(['128', '256-128', '512-256-128'], loc='lower right', prop={'size': 8})

    fig2.add_subplot(1, 3, 3)
    plt.plot(valid_acc128true)
    plt.plot(valid_acc128256true)
    plt.plot(valid_acc128256512true)
    plt.title('Valid acc for different hidden dims')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.xticks(np.arange(1, 21, 1))
    plt.legend(['128', '256-128', '512-256-128'], loc='lower right', prop={'size': 8})

    #plt.savefig('temp25.png', dpi=fig.dpi)

    plt.show()

    #######################
    # END OF YOUR CODE    #
    #######################



def lr_case():
    
    learning_rates = list(np.logspace(-6, 2, num=9))
    #print(learning_rates)
    hid_dim = [128]
    use_batch_norm = False
    batch_size = 128
    epochs = 10
    val_accs = []
    val_loss = []
    train_losses = []

    for lr in learning_rates:
        _ , _, _, logging_info = train_pytorch(hidden_dims = hid_dim, lr = lr, use_batch_norm = use_batch_norm, batch_size = batch_size , epochs = epochs, seed = 42, data_dir='data/')
        val_accs.append(max(logging_info['Valid accuracy']))
        val_loss.append(logging_info['Valid loss'])
        train_losses.append(logging_info['Train loss'])

    plt.plot(val_accs)
    plt.title('Val acc for different lr')
    plt.ylabel('Accuracy')
    plt.xlabel('Learning rate')
    plt.xscale('log')    
    plt.show()

    #plt.savefig('temp00.png', dpi=fig.dpi)


    for i in range(len(train_losses)):
        plt.plot(train_losses[i])
    plt.title('Train loss for different lr')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend([1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100], loc='center left', bbox_to_anchor=(1, 0.5))  
    plt.show()

    #plt.savefig('temp01.png', dpi=fig.dpi)

    for i in range(len(val_loss)):
        plt.plot(val_loss[i])
    plt.title('Valid loss for different lr')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend([1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100], loc='center left', bbox_to_anchor=(1, 0.5))  
    plt.show()

    #plt.savefig('temp01.png', dpi=fig.dpi)


    return

if __name__ == '__main__':
    # Feel free to change the code below as you need it.
    FILENAME = 'results2.json' 
    if not os.path.isfile(FILENAME):
        train_models(FILENAME)
    plot_results(FILENAME)
    lr_case()