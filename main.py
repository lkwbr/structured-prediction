#!/usr/bin/env python

# Luke Weber, 11398889
# CptS 580, HW #1
# Created 02/08/2017

import numpy as np
import string
import re

# Global
labels = list(string.ascii_lowercase)

def main():
    """ Setup for both handwriting and text-to-speech mapping problems """

    # Perceptron training params
    phi = lambda x, y: x**2 # TODO: unary features
    R = 20
    eta = 0.01
    MAX = 100

    # Raw training and testing data
    data_dir = "data/"
    raw_train_test = [(data_dir + "nettalk_stress_train.txt",
                       data_dir + "nettalk_stress_test.txt"),
                      (data_dir + "ocr_fold0_sm_train.txt",
                       data_dir + "ocr_fold0_sm_test.txt")]

    for raw_train, raw_test in raw_train_test:
        # Parse train & test data
        train = parse_data_file(raw_train)
        test = parse_data_file(raw_test)

        # Train structured perceptron!
        w = ospt(train, phi, R, eta, MAX)

        # Test
        # TODO
            
def parse_data_file(file_loc):
    """ Parse raw data into form of [(x_0, y_0), ..., (x_n, y_n)] """

    data_arr = []
    
    with open(file_loc) as f:
        for line in f:
            # Skip empty lines
            if not line.strip(): continue

            l_toks = line.split("\t")
            x = l_toks[1][2:] # Trim leading "im" tag
            y = l_toks[2]
            
            data_arr.append((x, y))

    return data_arr

def get_score(y_hat, w, x):
    pass

def get_random_y():
    pass

def get_max_one_char(y_hat, w, x):
    """
    Make one-character changes to y_hat, finding which
    single change produces the best score
    """

    # Initialize variables to max
    s_max = get_score(y_hat, w, x)
    y_max = y_hat

    for i in range(len(y_hat)):

        # Copy of y_hat to play with
        y_temp = copy.deepcopy(y_hat)

        # Go through a-z at i-th index
        for c in labels:
            y_temp[i] = c
            s_new = score(y_temp)
            if s_new > s_max:
                s_max = s_new
                y_max = y_temp
    
    return y_max

def rgs(x, phi, w, R):
    """ Randomized Greedy Search (RGS) inference  """

    for i in range(R):

        # Initialize best scoring output randomly
        y_hat = get_random_y()

        # Until convergence
        while True:
            y_max = get_max_one_char(y_hat)
            if y_max == y_hat: break
            y_hat = y_max

    return y_hat

def ospt(D, phi, R, eta, MAX):
    """ Online structured perceptron training """
    
    # Setup weights of scoring function to 0
    w = 0 # TODO

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

# Party = started
main()
