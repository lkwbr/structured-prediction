# recurrent.py

"""
RECURRENT CLASSIFIER
Reduction of strucutred prediction to multi-label classification; uses methods
like Immitation Learning [1] and the DAgger algorithm [2]; we use results from
the paper Structured Prediction via Output Space Search [3] for comparison

[1] http://proceedings.mlr.press/v24/vlachos12a/vlachos12a.pdf
[2] https://www.ri.cmu.edu/pub_files/2011/4/Ross-AISTATS11-NoRegret.pdf
[3] http://jmlr.org/papers/volume15/doppa14a/doppa14a.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import random

from util import *
from model import Model

class RecurrentClassifier(Model):
    """
    Reduction of structured prediction to multi-label classification
    """

    __name__ = "RecurrentClassifier"

    def __init__(self, alphabet, len_x):

        # NOTE: This alphabet update needs to happen before we do anything else,
        # mainly because of len_y's dependency
        self._dummy_label = "$"
        alphabet.update(self._dummy_label)

        # Have parent do their thing
        super().__init__(alphabet, len_x)

        # L: Set of classification examples
        self._L = []
        self._policy = self.LearnedPolicy(self._L)
        self._phi_pairs = []

    class Policy:
        __name__ = "AbstractPolicy"
        def __init__(self, L): self._L = L
        def action(self, f):
            """
            For the given feature input f, produce the right action (i.e. label)
            """
            raise NotImplementedError()

    class LearnedPolicy(Policy):

        __name__ = "LearnedPolicy"

        def __init__(self, L):
            super().__init__(L)
            self._classifier = svm.SVC(kernel = "linear")
            if len(self._L) != 0: self.learn()

        def action(self, f):
            return self._classifier.predict([f])[0]

        def learn(self, L = None):
            # (Possibly redundantly) update example set
            if L is not None: self._L = L
            else: L = self._L
            X = [t[0] for t in L]
            Y = [t[1] for t in L]
            self._classifier.fit(X, Y)

    class OraclePolicy(Policy):
        __name__ = "OraclePolicy"
        def __init__(self, L): super().__init__(L)
        def action(self, f):
            for lf, label in self._L:
                if np.array_equal(f, lf): return label

    @overrides(Model)
    def train(self, D, *args):
        raise NotImplementedError("Class method 'train' has not been implemented")

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

                if len(y_construct) == 0: y_current = [self._dummy_label]
                else: y_current = y_construct

                # Partial output features
                f = self._phi(x, y_current)

                # Predict
                y_hat = self._policy.action(f)
                y_construct.append(y_hat)

            # TODO: Compare Hamming accuracy here (recurrent error?)
            num_wrong_labels = list_diff(y, y_construct)
            hamming_loss += num_wrong_labels / len(y)

            #print(give_err_bars(self._alphabet, y, y_construct), flush = True)

        hamming_accuracy = (1.0 - (hamming_loss / len(D)))
        print("Done: Hamming accuracy is {}%".format(round(hamming_accuracy * 100)))
        print()

        return hamming_accuracy

    # NOTE: Single underscore is used to prevent "name mangling", which will now
    # allow inhereting classes to use this method as their own
    def _generate_examples(self, D):
        """ Populate list of classification examples """

        # Through each training example
        for x, y in D[:]:

            # Add training example for each left-anchored subset of the
            # structured output y; (left-anchored subsets: "h", "he", "hey")
            T = len(y)
            for t in range(T):

                y_t = y[t]
                y_partial = y[:t]
                if t == 0: y_partial = [self._dummy_label]

                # Generate features, package up, and append to list
                f = self._phi(x, y_partial)
                example = (f, y_t)
                self._L.append(example)

    # TODO: Possibly extract out this __phi func to Model class
    def _phi(self, x, y):
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
        self._phi_pairwise_base_index = self._len_x * self._len_y
        dimen = (self._len_x * self._len_y) + (self._len_y ** 2)

        features = np.zeros((dimen))
        alpha_list = list(self._alphabet)
        alpha_list.sort()

        # (One-time) Generate pair-index object
        if len(self._phi_pairs) == 0:
            for a in alpha_list:
                for b in alpha_list:
                    p = a + b
                    self._phi_pairs.append(p)

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
            comb_index = self._phi_pairs.index(p)
            vect_index = self._phi_pairwise_base_index + comb_index

            # Update occurace of pair
            features[vect_index] += 1

        return features

class ImitationClassifier(RecurrentClassifier):

    __name__ = "ImitationRecurrentClassifier"

    def __init__(self, alphabet, len_x):
        super().__init__(alphabet, len_x)

    @overrides(RecurrentClassifier)
    def train(self, D, *args):

        print("Generating examples...", flush = True)
        self._generate_examples(D)

        # Train SVM on our generated examples
        print("Training classifier...", flush = True)
        self._policy.learn(self._L)

        # NOTE: Attempted custom, point-by-point training with the library
        # classifier; but, this gave worse accuracy

class DAggerClassifier(RecurrentClassifier):

    __name__ = "DAggerRecurrentClassifier"

    def __init__(self, alphabet, len_x, beta):
        super().__init__(alphabet, len_x)
        self._beta = beta # Interpolation parameter
        self.policy = None

    @overrides(RecurrentClassifier)
    def train(self, D, *args):

        # NOTE: Would likely be more efficient to store L as a dict

        # Splitup training and validation data, 75/25
        # IDEA: Could shuffle D to ensure uniform distribution
        v_split = int(0.75 * len(D))
        train_data = D[:v_split]
        validation_data = D[v_split:]

        print("Generating examples...", flush = True)
        self._generate_examples(train_data)

        # Train SVM on our generated examples
        print("Training classifier...", flush = True)

        # Instantiate initial classifier
        h_learned = self.LearnedPolicy(self._L)
        h_oracle = self.OraclePolicy(self._L)
        h_history = [] # History of learned classifiers through iterations

        # Data aggregation iterations
        d_max = args[0]
        for j in range(d_max):

            h_current = self._choose_policy(h_oracle, h_learned)
            print("\t", j + 1, d_max, h_current.__name__, flush = True)
            for x, y in train_data[:]:

                # Iterated select partial outputs of y
                T = len(y)
                for t in range(T):

                    # Generate features
                    y_partial = y[:t]
                    if t == 0: y_partial = [self._dummy_label]
                    f = self._phi(x, y_partial)

                    # Does current classifier match oracle classification?
                    c_action = h_current.action(f)  # Predicted action
                    o_action = h_oracle.action(f)   # Oracle action
                    if c_action == o_action:

                        # Add classification example (from oracle)
                        # TODO: Describe why we do this! Don't we already have
                        # this classification example in L?
                        clf_example = (f, o_action)
                        self._L.append(clf_example)

            # Store history of all learned policies, as well as Learn a new
            # classifier from aggregate data!
            if h_current is not h_oracle: h_history.append(h_learned)
            h_learned = self.LearnedPolicy(self._L)

        # Find best learned classifier based on validation data
        h_best_tup = (None, -1)
        for h in h_history:
            self._policy = h
            acc = self.test(validation_data)
            if acc > h_best_tup[1]: h_best_tup = (h, acc)

        # Set our policy to the max learned classifier
        self._policy = h_best_tup[0]
        return h_best_tup[0]

    def _choose_policy(self, a, b):
        """
        Choose policy a with probability beta, and policy b with probability
        (1 - beta)
        """
        choice_point = random.random()
        if choice_point < self._beta: return a
        return b
