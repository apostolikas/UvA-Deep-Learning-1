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
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from collections import OrderedDict


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes, use_batch_norm=False):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP
          use_batch_norm: If True, add a Batch-Normalization layer in between
                          each Linear and ReLU layer.

        TODO:
        Implement module setup of the network.
        The linear layer have to initialized according to the Kaiming initialization.
        Add the Batch-Normalization _only_ is use_batch_norm is True.
        
        Hint: No softmax layer is needed here. Look at the CrossEntropyLoss module for loss calculation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        #pass
        
        super(MLP, self).__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.flag = 0

        self.fc_layers = nn.ModuleList() 
        self.bn_layers = nn.ModuleList()
        self.elus = nn.ModuleList()

        for j in range(len(n_hidden)+1):
            if j==0:
                self.fc_layers.append(nn.Linear(n_inputs, n_hidden[0]))
                if use_batch_norm:
                    self.flag = 1
                    self.bn_layers.append(nn.BatchNorm1d(n_hidden[0]))
            elif j == len(n_hidden):
                self.fc_layers.append(nn.Linear(n_hidden[j-1], n_classes))
            else:
                self.fc_layers.append(nn.Linear(n_hidden[j-1], n_hidden[j]))
                if use_batch_norm:
                    self.flag = 1
                    self.bn_layers.append(nn.BatchNorm1d(n_hidden[j]))
            self.elus.append(nn.ELU())

        for layer in self.fc_layers:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in')


        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        out = x.view(x.shape[0], -1)

        for j in range(len(self.n_hidden)+1):
            out = self.fc_layers[j].forward(out)
            if j != len(self.n_hidden):
                if self.flag == 1: 
                    out = self.bn_layers[j].forward(out)    
                out = self.elus[j].forward(out) 

        
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device
    
