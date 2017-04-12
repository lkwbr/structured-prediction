# recurrent.py

"""
RECURRENT CLASSIFIER
xxx
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

from util import *
from model import Model

class ImmitationClassifier(RecurrentClassifier):

    __name__ = "ImmitationRecurrentClassifier"

class RecurrentClassifier(Model):
    """
    Reduction of structured prediction to multi-label classification
    """

    __name__ = "RecurrentClassifier"

    def __init__(self, alphabet, len_x):

        # Handle dummy label
        self.__dummy_label = "$"
        alphabet.update(self.__dummy_label)

        # Have parent do their thing
        super().__init__(alphabet, len_x)

        # L: Set of classification examples
        self.__L = []
        self.__classifier = svm.SVC(kernel = "linear")
        self.__phi_pairs = []

    @overrides(Model)
    def train(self, D):

        print("Generating examples...", flush = True)
        self.__generate_examples(D)

        # Train SVM on our generated examples
        print("Training classifier...", flush = True)
        X = [t[0] for t in self.__L]
        Y = [t[1] for t in self.__L]
        self.__classifier.fit(X, Y)

        # NOTE: Attempted custom, point-by-point training with the library
        # classifier; but, this gave worse accuracy

    @overrides(Model)
    def test(self, D):

        # TODO: Compute recurrent error
        # TODO: Compute oracle error

        print("Testing classifier...", flush = True)
        hamming_loss = 0
        for x, y in D[:]:

            T = len(y)
            y_construct = []
            for t in range(T):

                if len(y_construct) == 0: y_current = [self.__dummy_label]
                else: y_current = y_construct

                # Partial output features
                f = self.__phi(x, y_current)

                # Predict
                y_hat = self.__classifier.predict([f])[0]
                y_construct.append(y_hat)

            # TODO: Compare Hamming accuracy here (recurrent error?)
            num_wrong_labels = list_diff(y, y_construct)
            hamming_loss += num_wrong_labels / len(y)

            print(give_err_bars(self.alphabet, y, y_construct), flush = True)

        hamming_accuracy = (1.0 - (hamming_loss / len(D)))

        print("Done: Hamming accuracy is {}%".format(round(hamming_accuracy * 100)))
        print()

        return hamming_accuracy

    def __generate_examples(self, D):
        """ Populate list of classification examples """

        # Through each training example
        for x, y in D[:]:

            # Add training example for each left-anchored subset of the
            # structured output y; (left-anchored subsets: "h", "he", "hey")
            T = len(y)
            for t in range(T):

                y_t = y[t]
                y_partial = y[:t]
                if len(y_partial) == 0: y_partial = [self.__dummy_label]

                #print(y_partial, y_t)

                # Generate features, package up, and append to list
                f = self.__phi(x, y_partial)
                example = (f, y_t)
                self.__L.append(example)

    # TODO: Possibly extract out this __phi func to Model class
    def __phi(self, x, y):
        """
        First-order joint-feature function, phi:
            0. (Unary) Do unary features
            1. (Pairwise) Capture all two-char permutations
            2. Assign these permuations consistent indices
            3. Count frequencies of each permuation and update vector
               at that index

        Supports generating features for partially labelled outputs, i.e.,
        for situations where len(y) <= len(x)
        """

        # Initial setting of phi dimensions
        self.__phi_pairwise_base_index = self.len_x * self.len_y
        dimen = (self.len_x * self.len_y) + (self.len_y ** 2)

        features = np.zeros((dimen))
        alpha_list = list(self.alphabet)
        alpha_list.sort()

        # (One-time) Generate pair-index object
        if len(self.__phi_pairs) == 0:
            for a in alpha_list:
                for b in alpha_list:
                    p = a + b
                    self.__phi_pairs.append(p)

        # Unary features
        for i in range(len(y)):

            x_i = x[i]
            y_i = y[i]

            # Unary features
            index = alpha_list.index(y_i)
            x_vect = np.array(x_i)
            y_target = len(x_i) * index
            for j in range(len(x_i)): features[j + y_target] += x_vect[j]

        # Pairwise features
        for i in range(len(y) - 1):

            # Get pair index
            a = y[i]
            b = y[i + 1]
            p = a + b
            comb_index = self.__phi_pairs.index(p)
            vect_index = self.__phi_pairwise_base_index + comb_index

            # Update occurace of pair
            features[vect_index] += 1

        return features
