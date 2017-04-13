#!/usr/bin/env python3

# Luke Weber, 11398889
# CptS 580, HW #3
# Created 02/08/2017
# Last modified 04/04/2017

# TODO: Abstract every model to type Model
# TODO: Unify main.py for many different SP classifiers
# TODO: Add description and link to LaSO paper with beam search
# TODO: Add description and link to recurrent classification
# TODO: Abstract out the plotter
# TODO: Upgrade the damn readme.md

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
    - Fourth-order
These feature representations allow us to create different
features for each structured input x and output y.
"""

# Third-party libraries
import numpy as np
import signal
import string
import math
import time
import sys
import re
import os

# Custom classes and definitions
from util import *
from perceptron import StructuredPerceptron
from recurrent import ImitationClassifier, DAggerClassifier

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

def run_sp(raw_train_test, update_limit, b_limit, data_limit, report_name):
    """
    Run: Training function
    Data: Handwritten words and text-to-speech
    Setup (for both handwriting and text-to-speech mapping) problems:
        0. Establish model parameters
        1. Parse training and testing data
        2. Train structured perceptron on training data
        3. Test on testing data
    Assumptions:
        0. All data in form "000001010101010101..." "[label]"
    """

    stat_reports = []

    # Perceptron training params
    # NOTE: Tune these according to your preference
    R = [10, 25, 50, 100, 200][1]
    eta = [0.01][0]                     # Learning rate
    MAX = [50][0]                       # Maximum number of iterations
    phi_order = [2][0]                  # Joint-features: 1 = unary, etc.
    bs = [1, 5, 10, 15, 25, 50, 100]    # Beam width
    search_types = [0, 1]               # Search type: Best-first or breadth-first
    update_methods = [0, 1, 2]          # Updates: standard, early update, max-violation
    load_w = [True, False][1]           # Determines loading model weights

    # First: Run on breadth-first and best-first
    for search_type in search_types[1:]:
        for update_method in update_methods[update_limit[0]:update_limit[1]]:
            for b in bs[b_limit[0]:b_limit[1]]:
                for raw_train, raw_test in raw_train_test[data_limit[0]:data_limit[1]]:

                    # Let's time parsing, training, and testing
                    start_time = time.clock()

                    print("Parsing training and testing data:")
                    print("\t" + raw_train)
                    print("\t" + raw_test)

                    # Parse train & test data
                    train, len_x, alphabet = parse_data_file(raw_train)
                    test, *_ = parse_data_file(raw_test)

                    # Initialize model with parameters
                    sp = StructuredPerceptron(alphabet, len_x, phi_order,
                        update_method, search_type, R, eta, MAX, b)

                    # Train & test
                    train_accuracy = sp.train(train)
                    test_accuracy = sp.test(test)

                    # Collect and record stats
                    # NOTE: Format ([data ID], [train accuracy], [test accuracy],
                    # [elapsed time], [search type], [update method],
                    # [beam width], [end date-time])
                    elapsed_time = (round(time.clock() - start_time) / 60)
                    end_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                    report = (raw_train, train_accuracy, test_accuracy,
                        elapsed_time, search_type, update_method, b, end_datetime)
                    stat_reports.append(report)

                    # Write report to file
                    write_report(report, report_name)

def run_rc(raw_train_test):
    """ Train and test our recurrent classifiers """

    for raw_train, raw_test in raw_train_test[1:]:

        # Parse train & test data
        print("Parsing training/testing data...", flush = True)
        train, len_x, alphabet = parse_data_file(raw_train)
        test, *_ = parse_data_file(raw_test)

        # Construct
        #ic = ImitationClassifier(alphabet, len_x)
        dc = DAggerClassifier(alphabet, len_x, beta = 0.5)

        # Train and test
        #h_ic = ic.train(train)
        h_dc = dc.train(train, 5)
        #accuracy_ic = ic.test(test)
        accuracy_dc = dc.test(test)

def main():
    """ Driver function """

    # Get our raw training and testing data
    data_dir = "data/"
    raw_train_test = get_data_files(data_dir)

    run_rc(raw_train_test)

    # FIXUPS (from program crash)
    # [1] Standard; beam = 50, 100; data = OCR
    #run((0, 1), (5, 7), (1, 2), "report_breadth_standard")
    # [2] Early; beam = 25; data = OCR
    #run((1, 2), (4, 5), (1, 2), "report_breadth_early")
    # [2] Max-violation; beam = 15, 25; data = OCR
    #run((2, 3), (3, 5), (1, 2), "report_breadth_max")

    # PROGRESS
    # [3] Early; beam = 50, 100; data = Nettalk, OCR
    # run_sp((1, 2), (5, 7), (0, 2), "report_breadth_early")
    # [3] Max-violation; beam = 50, 100; data = Nettalk, OCR
    # run_sp((2, 3), (5, 7), (0, 2), "report_breadth_max")

# Party = started
main()
