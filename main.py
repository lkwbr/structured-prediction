#!/usr/bin/env python3

# Luke Weber, 11398889
# CptS 580, HW #1
# Created 02/08/2017

import pyqtgraph as pg
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
verbose = False
alphabet = set()
len_x = -1
phi_dimen = -1

def main():
    """
    Setup for both handwriting and text-to-speech mapping problems:
        0. Establish model parameters
        1. Parse training and testing data
        2. Train structured perceptron on training data
        3. Test on testing data

    Assumptions:
        0. All data in form "000001010101010101..." "[label]"
    """

    global len_x, phi_dimen

    # Perceptron training params
    R = 20
    eta = 0.01
    MAX = 100

    # Raw training and testing data
    data_dir = "data/"
    raw_train_test = [(data_dir + "nettalk_stress_train.txt",
                       data_dir + "nettalk_stress_test.txt")]
                      #(data_dir + "ocr_fold0_sm_train.txt",
                       #data_dir + "ocr_fold0_sm_test.txt")]

    for raw_train, raw_test in raw_train_test:

        print("Parsing data...")
        
        # Parse train & test data
        train, len_x, len_y = parse_data_file(raw_train)
        #test = parse_data_file(raw_test)

        # From data -> joint feature function
        phi_dimen = len_x * len_y
        phi = phi_func
        
        # Train structured perceptron!
        w = ospt(train, phi, R, eta, MAX)

        # Test
        # TODO

def dprint(s):
    if verbose: print(s)

def phi_func(x, y):
    """ Joint-feature function """

    vect = np.zeros((phi_dimen))

    for i in range(len(x)):

        x_i = x[i]
        y_i = y[i]

        alpha_list = list(alphabet)
        # Sorting keeps consistency of indices with respect to
        # all prior and following phi(x, y) vectors
        alpha_list.sort() 
        index = alpha_list.index(y_i)
        x_vect = np.array(x_i)

        # Manual insertion of x into standard vector
        y_target = len(x_i) * index
        for j in range(len(x_i)): vect[j + y_target] = x_vect[j]
    
    return vect
            
def parse_data_file(file_loc):
    """ Parse raw data into form of [(x_0, y_0), ..., (x_n, y_n)] """

    global alphabet

    data_arr = []
    len_x_vect = -1
    
    with open(file_loc) as f:

        x = []
        y = []

        # Take collection of examples (e.g. collection of pairs of
        # character data x_i matched with the actual character
        # class y_i) and push into data array
        for line in f:

            # Push single collection of examples onto data array
            # when newline is encountered
            if not line.strip():
                data_arr.append((x, y))
                x = []
                y = []
                continue

            # Parse one example (x_i, y_i)
            l_toks = line.split("\t")
            x_i_str = l_toks[1][2:] # Trim leading "im" tag
            x_i = [int(c) for c in x_i_str]
            y_i = setify(l_toks[2])

            # Collect length of all x_i's
            if len_x_vect < 0: len_x_vect = len(x_i)

            # Take note of all possible labels (i.e. the set Y)
            # NOTE: listifying y_i is necessary to keep leading
            # zeroes, e.g. maintaining '04' rather than '4'
            alphabet.update([y_i])

            # Add single example to collection
            x.append(x_i)
            y.append(y_i)

    num_labels = len(alphabet)
            
    return data_arr, len_x_vect, num_labels

def setify(num):
    """
    Set-ify that number (i.e. remove trailing zeros, as is
    automatically done in the alphabet set) for consistency
    """
    
    return list(set([num]))[0]

def get_score(w, phi, x, y_hat):
    return np.dot(w, phi(x, y_hat))

def get_random_y(len_y):

    rand_y = []

    # If no length passed
    if len_y == None:
        min_word_len = 2
        max_word_len = 6
        rand_word_len = math.floor(random.uniform(min_word_len,
                                                  max_word_len + 1))
    else: rand_word_len = len_y

    for i in range(rand_word_len):

        rand_char = random.sample(alphabet, 1)[0]
        rand_y.append(rand_char)
    
    return rand_y

def get_max_one_char(w, phi, x, y_hat):
    """
    Make one-character changes to y_hat, finding which
    single change produces the best score; we return
    that resultant y_max
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

def rgs(x, phi, w, R, len_y):
    """
    Randomized Greedy Search (RGS) inference:
    Try and use the current weights to arrive at
    the correct label; we will always return our
    best guess
    """

    for i in range(R):

        # Initialize best scoring output randomly
        y_hat = get_random_y(len_y)
        #print(y_hat)

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

    # Record model's progress w.r.t. accuracy
    acc_progress = []
    pw = pg.plot()
    
    # Setup weights of scoring function to 0
    w = np.zeros((phi_dimen))

    # Iterate until max iterations or convergence
    # TODO: Check for convergence
    for it in range(MAX):

        print("[Iteration " + str(it) + "]\n")

        train_num = 0
        num_mistakes = 0
        num_correct = 0
        
        # Go through training examples
        # TODO: Remove limitation
        for x, y in D[:100]:

            print("\tTraining example #" + str(train_num) + "...")
            
            # Predict
            # NOTE: Passing in len(y) so we know what kind of
            # y_hat to generate randomly at the start
            y_hat = rgs(x, phi, w, R, len(y))

            dprint("y = " + str(y))
            dprint("y_hat = " + str(y_hat))

            # If error, update weights
            if y_hat != y:
                dprint("Mistake: Updating weights!")
                w = np.add(w, np.dot(eta, (np.subtract(phi(x, y), phi(x, y_hat)))))
                num_mistakes += 1
            else:
                dprint("Good: Predicted correctly!")
                num_correct += 1

            train_num += 1

        # Determine accuracy
        accuracy = num_correct / (num_correct + num_mistakes)
        acc_progress.append(accuracy)

        # Plot accuracy timline
        if len(acc_progress) > 1:
            pw.plot(acc_progress, clear=True)
            pg.QtGui.QApplication.processEvents()

        # Report iteration stats
        print()
        print("| Accuracy = " + str(accuracy))
        print("| Number correct = " + str(num_correct))
        print("| Number of mistakes = " + str(num_mistakes))
        print()
    
    return w

# Party = started
main()
