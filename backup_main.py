#!/usr/bin/env python3

# Luke Weber, 11398889
# CptS 580, HW #1
# Created 02/08/2017

import numpy as np
import random
import string
import math
import copy
import re

# TODO:
#   - Get RGS and training working with just one character!
#   - Start parsing entire word from data (not just character)
#   - Alter phi function drastically to accompany word

# Globals
#Y = [] # Set of all possible y's
alphabet = set()
len_x = -1
len_Y = -1
phi_dimen = -1

def main():
    """
    Setup for both handwriting and text-to-speech mapping problems:
        0. Establish model parameters
        1. Parse training and testing data
        2. Train structured perceptron on training data
        3. Test on testing data

    Assumptions:
        0. All data in form "000001010101010101" "label"
    """

    global len_x, len_Y, phi_dimen

    # Perceptron training params
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
    
        # From data -> joint feature function
        #Y = alphabet #list(set([y for x, y in train]))
        len_x = len(train[0][0])
        len_Y = len(alphabet)

        phi_dimen = len_x * len_Y
        phi = phi_func

        # Train structured perceptron!
        w = ospt(train, phi, R, eta, MAX)

        # Test
        # TODO

def phi_func(x, y):
    """ Joint-feature function """

    vect = np.zeros((phi_dimen))
    # NOTE: Depending on list-ification of set
    # to provide consistent indices for each element
    index = list(alphabet).index(y[0])
    x_vect = np.array(x)

    # Manual insertion of x into standard vector
    y_target = len(x) * index
    for i in range(len(x)): vect[i + y_target] = x_vect[i]
    
    return vect
            
def parse_data_file(file_loc):
    """ Parse raw data into form of [(x_0, y_0), ..., (x_n, y_n)] """

    global alphabet

    data_arr = []
    
    with open(file_loc) as f:
        for line in f:
            # Skip empty lines
            if not line.strip(): continue

            l_toks = line.split("\t")
            x_str = l_toks[1][2:] # Trim leading "im" tag
            x = [int(c) for c in x_str]
            y = [l_toks[2]]
            # Set-ify that number (i.e. remove trailing zeros, as
            # is automatically done in the alphabet set) for
            # consistency
            y[0] = list(set(y[0]))[0]
            alphabet.update(l_toks[2])
            
            data_arr.append((x, y))

    return data_arr

def get_score(w, phi, x, y_hat):
    return np.dot(w, phi(x, y_hat))

def get_random_y():
    return random.sample(alphabet, 1)

def get_max_one_char(w, phi, x, y_hat):
    """
    Make one-character changes to y_hat, finding which
    single change produces the best score
    """

    # Initialize variables to max
    s_max = get_score(w, phi, x, y_hat)
    y_max = y_hat

    for i in range(len(y_hat)):

        # Copy of y_hat to play with
        y_temp = copy.deepcopy(y_hat)

        # Go through a-z at i-th index
        for c in alphabet:
            
            y_temp[i] = c
            s_new = get_score(w, phi, x, y_temp)
            if s_new > s_max:
                s_max = s_new
                y_max = y_temp
    
    return y_max

def rgs(x, phi, w, R):
    """ Randomized Greedy Search (RGS) inference  """

    for i in range(R):

        # Initialize best scoring output randomly
        y_hat = get_random_y()
        print(y_hat)

        # Until convergence
        while True:
            y_max = get_max_one_char(w, phi, x, y_hat)
            if y_max == y_hat: break
            y_hat = y_max

    return y_hat

def ospt(D, phi, R, eta, MAX):
    """ Online structured perceptron training """

    print("Training Structured Perceptron:")
    print()
    print("\tData length = " + str(len(D)))
    print("\tNumber of restarts = " + str(R))
    print("\tLearning rate = " + str(eta))
    print("\tMax iteration count = " + str(MAX))
    print()
    
    # Setup weights of scoring function to 0
    w = np.zeros((phi_dimen))

    # Iterate until max iterations or convergence
    # TODO: Check for convergence
    for it in range(MAX):

        print("[Iteration " + str(it) + "]")

        num_mistakes = 0
        num_correct = 0
        
        # Go through training examples
        for x, y in D:

            #print("Running Randomized Greedy Search...")
            
            # Predict
            y_hat = rgs(x, phi, w, R) 

            # Check error
            error = y_hat != y

            #print("\ty = " + str(y))
            #print("\ty_hat = " + str(y_hat))

            # If error, update weights
            if error:
                #print("Mistake: Updating weights!")
                w = np.add(w, np.dot(eta, (np.subtract(phi(x, y), phi(x, y_hat)))))
                num_mistakes += 1
            else:
                #print("Good: Predicted correctly!")
                num_correct += 1

        # Report iteration stats
        print("Number correct = " + str(num_correct))
        print("Number of mistakes = " + str(num_mistakes))
        #print(w)
        return
    
    return w

# Party = started
main()
