#!/usr/bin/env python3

# Luke Weber, 11398889
# CptS 580, HW #2
# Created 02/08/2017
# Last edited 03/13/2017

"""
Structured Perceptron, predicting structured outputs y_hat for
structured inputs x with the following inference (and update)
methods:
    - Randomized Greedy Search (RGS)
    - Standard Best-First Beam Search (BSTFBS)
    - Standard Breadth-First Beam Search (BDTFBS)
    - Early Update with BSTFBS
    - Max-Violation with BSTFS
And on top of all this we offer a variety of joint-feature
representations, such as:
    - Unary
    - Pairwise / First-order
    - Third-order
    - Fourth-order.
These feature representations allow us to create
different features for each structured input x and output y.
"""

# Third-party libraries
import numpy as np
import signal
import string
import math
import sys
import re
import os

# Custom classes and definitions
from util import *
from model import StructuredPerceptron

# Running on Windows?
if os.name != "posix":
    windows = True
    import winsound
else: windows = False

# Debugging
verbose = False         # Debug output control
sig = False             # See if a signal is already being handled

# Scoring function (weights)
weights = []
weights_dir = "weights/"

def main():
    """
    Main: Driver function
    Data: Handwritten words and text-to-speech
    Setup (for both handwriting and text-to-speech mapping) problems:
        0. Establish model parameters
        1. Parse training and testing data
        2. Train structured perceptron on training data
        3. Test on testing data
    Assumptions:
        0. All data in form "000001010101010101..." "[label]"
    """

    # Save weights on Ctrl-C
    signal.signal(signal.SIGINT, signal_handler)

    # Set non-truncated printing of numpy arrays
    np.set_printoptions(threshold = np.inf)

    # Perceptron training params
    # NOTE: Tune these accordingly
    R = [10, 25, 50, 100, 200][1]
    eta = 0.01
    MAX = 50
    phi_order = 2   # e.g. 1 = unary, 2 = pairwise, 3 = third-order, etc.
    b = 5           # Beam width
    mi = 2          # Method index
    load_w = False

    # Raw training and testing data
    data_dir = "data/"
    raw_train_test = get_data_files(data_dir)

    for raw_train, raw_test in raw_train_test[1:]:

        print("Parsing training and testing data:")
        print("\t" + raw_train)
        print("\t" + raw_test)

        # Parse train & test data
        train, len_x, alphabet = parse_data_file(raw_train)
        test, *_ = parse_data_file(raw_test)

        # Initialize model with parameters
        sp = StructuredPerceptron(alphabet, len_x, phi_order, mi, R, eta, MAX, b)

        # NOTE: We can either train for weights, or load them
        if load_w: w = load_w()
        else: w = sp.train(train)

        # Test
        accuracy = sp.test(test)

    return

# Party = started
main()
