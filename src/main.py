#!/usr/bin/env python3

# Luke Weber, 11398889
# CptS 580, HW #2
# Created 02/08/2017

"""
Structured Perceptron with Randomized Greedy Search to
make inferences; running with varying joint-feature
representations, such as: unary, pairwise, third-order,
and fourth-order.

NOTE: Currently, we have a problem with this program not
converging, which is likely due to a small, language-
dependent error.
"""

from model import *
from util import *

import pyqtgraph as pg
import numpy as np
import random
import signal
import string
import math
import time
import copy
import sys
import re
import os

# Running on Windows?
if os.name != "posix":
    windows = True
    import winsound
else: windows = False

# Debugging
verbose = False         # Debug output control
sig = False             # See if a signal is already being handled

# Model
alphabet = set()        # Set of given dataset's alphabet of labels

# Phi-related
len_x = -1
phi_dimen = -1
pairwise_base_index = -1
triplet_base_index = -1
quadruplet_base_index = -1
pairs = []
triplets = []
quadruplets = []

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

    global len_x, phi_dimen

    # Save weights on Ctrl-C
    signal.signal(signal.SIGINT, signal_handler)

    # Set non-truncated printing of numpy arrays
    np.set_printoptions(threshold = np.inf)

    # Perceptron training params
    R = [10, 25, 50, 100, 200][1]
    eta = 0.01
    MAX = 100
    L = 100 #None

    # Raw training and testing data
    data_dir = "data/"
    raw_train_test = [(data_dir + "nettalk_stress_train.txt",
                       data_dir + "nettalk_stress_test.txt"),
                      (data_dir + "ocr_fold0_sm_train.txt",
                       data_dir + "ocr_fold0_sm_test.txt")]
    data_limit = 1 #len(raw_train_test)

    for raw_train, raw_test in raw_train_test[:data_limit]:

        print()
        print("Parsing training and testing data:")
        print("\t" + raw_train)
        print("\t" + raw_test)

        # Parse train & test data
        train, len_x, len_y = parse_data_file(raw_train)
        test, *_ = parse_data_file(raw_test)

        print("Done parsing!")
        print()

        # From data -> joint feature function; detect and
        # set phi_dimen dynamically
        phi = [phi_unary, phi_pairwise, phi_third_order, phi_fourth_order][3]
        phi_dimen = len(phi(train[0][0], train[0][1], len_x, len_y))

        # NOTE: We can either train for weights, or load them
        load_w = False
        if load_w: w = load_w()
        else: w = ospt(train, phi, R, eta, MAX, L)

        # Test
        ospt(test, phi, R, 1, 1, L, w)

        # Clear proverbial canvas
        reset_data_vars()

    return

# Party = started
main()
