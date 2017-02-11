#!/usr/bin/env python

# Luke Weber, 11398889
# CptS 580, Jana Doppa
# Created 02/08/2017

import numpy as np

def main():
    
    # TODO: Setup for both handwriting and text-to-speech mapping problem

    pass

def rgs(x, phi, w, R):
    """ Randomized Greedy Search (RGS) inference  """

    # Initialize best scoring output randomly
    y_hat = 0

    return y_hat 

def ospt(D, phi, R, eta, MAX):
    """ Online structured perceptron training """
    
    # Setup weights of scoring function to 0
    w = 0

    # Iterate until max iterations or convergence
    for it in range(MAX):
        # Go through training examples
        for x, y in D:
            # Predict
            y_hat = rgs(x, phi, w, R) 

            # Check error
            error = (y_hat - y) != 0

            # If error, update weights
            if error:
                w = w + eta * (phi(x, y) - phi(x, y_hat))
    
    return w

main()
