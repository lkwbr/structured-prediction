#!/usr/bin/env python3

# Luke Weber, 11398889
# CptS 580, HW #3
# Created 02/08/2017
# Last modified 04/14/2017

# TODO: Unify main.py for many different SP classifiers
# TODO: Add description and link to LaSO paper with beam search
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
from data import Data
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

# TODO: Update this to new data object
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

def run_rc(data):
    """ Train and test our recurrent classifiers """

    # Get email stuff at beginning
    #email = input("Email address: ")
    #password = input("Email password: ")

    for dset in data.parsed[1:]:

        # Construct list of classifiers
        clfs = [ImitationClassifier(dset.alphabet, dset.len_x), \
                DAggerClassifier(dset.alphabet, dset.len_x, beta = 0.5)]

        # Run through each classifier
        for c in clfs[:1]:

            print("Dataset = {}".format(dset.__name__))

            # Train and test
            ts = time.time()
            train_acc = c.train(dset.train)
            test_acc = c.test(dset.test)

            # Show time elapsed
            time_elapsed = round(((time.time() - ts) / 60), 2) # minutes
            print("{} minutes elapsed".format(time_elapsed))

            # Send me an email update
            subject = "[CptS 580] {} on {}".format(c.__name__, dset.__name__)
            body = "train = {}%\ntest = {}%\ntime = {} min".format(\
                round(train_acc * 100, 2), round(test_acc * 100, 2), time_elapsed)

            #send_email(email, password, subject, body)
            #print("Email sent to {}\n".format(email))

def main():
    """ Driver function """

    # Parse training and testing data
    data = Data(data_dir = "data/")

    # Run recurrent classifier
    run_rc(data)

# Party = started
main()
