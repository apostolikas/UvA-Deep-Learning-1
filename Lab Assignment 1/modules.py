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
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization. 
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        self.params = {'weight':None, 'bias':None}
        if input_layer == True:
          self.params['weight'] = np.random.normal(0,np.sqrt(1/in_features) ,size= (in_features, out_features))
        else:
          self.params['weight'] = np.random.normal(0,np.sqrt(2/in_features) ,size= (in_features, out_features))
        self.params['bias'] = np.zeros((out_features,))
        self.grads =  {'weight':None, 'bias':None}
        
        
        
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        out = x @ self.params['weight'] + self.params['bias']
        self.y = x
        
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        
        dx = dout @ self.params['weight'].T 
        self.grads['weight'] = self.y.T @ dout
        self.grads['bias'] = np.sum(dout, axis=0)
        
       
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        #pass
        
        self.y = None
        self.grads['weight'] = None
        self.grads['bias'] = None
        
        #######################
        # END OF YOUR CODE    #
        #######################


class ELUModule(object):
    """
    ELU activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        self.x = x
        out = np.where(x <= 0, 1 * (np.exp(x) - 1), x)
        self.y = out
        
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        elugrad = np.where(self.x <= 0, 1 + self.forward(self.x), 1)
        dx = dout * elugrad
        
        
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        self.x = None
        self.y = None 

        #######################
        # END OF YOUR CODE    #
        #######################


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.x = x
        b = x.max(axis = 1,keepdims=True)
        numerator = np.exp(x - b)
        denominator = numerator.sum(axis = 1, keepdims=True)
        out = numerator/denominator
        self.y = out
        
        
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        dx = np.zeros((dout.shape[0],dout.shape[1]))
        for i in range(dx.shape[0]):
          for j in range(dx.shape[1]):
            dx[i,j] = self.y[i,j] * (dout[i,j] - np.sum(dout[i,...] * self.y[i,...])) # 1 row before the final written solution
            
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        self.x = None
        self.y = None

        #######################
        # END OF YOUR CODE    #
        #######################


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        N = y.shape[0]
        x[x < 1e-9] = 1e-9 
        yonehot = np.zeros((x.shape[0],x.shape[1]))
        yonehot[np.arange(y.size), y] = 1
        loss = (-1/N)*np.sum(yonehot*np.log(x))       
        out = loss

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        x[x < 1e-9] = 1e-9
        N = y.shape[0]
        yonehot = np.zeros((x.shape[0],x.shape[1]))
        yonehot[np.arange(y.size), y] = 1
        dx = (-1/N)*(yonehot/x)
        
        
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx