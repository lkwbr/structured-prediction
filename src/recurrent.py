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
import math

from util import *
from model import Model

class RecurrentClassifier(Model):
    """
    Reduction of structured prediction to multi-label classification
    """

    __name__ = "RecurrentClassifier"

    def __init__(self, alphabet, len_x, phi_order = 2):

        # Have parent do their thing
        super().__init__(alphabet, len_x, phi_order)

        # L: Set of classification examples
        self._L = []
        self._policy = self.LearnedPolicy(self._L)

    class Policy:

        __name__ = "AbstractPolicy"

        def __init__(self, L):
            """ Shuffle and keep given classification examples """
            self._L = L
            random.shuffle(self._L)

        def action(self, f):
            """
            For the given feature input f, produce the right action (i.e. label)
            """
            raise NotImplementedError("No 'action' specified")

        def score(self, L = None):
            """ Returns training accuracy in [0, 1] """
            raise NotImplementedError("No 'score' function specified")

        def decomp(self):
            """ Decompose classification set L """
            X = [t[0] for t in self._L]
            Y = [t[1] for t in self._L]
            return X, Y

    class LearnedPolicy(Policy):

        __name__ = "LearnedPolicy"

        def __init__(self, L):
            super().__init__(L)
            # NOTE: Gamma is the amount each training example affects the model
            self._classifier = svm.SVC(kernel = "linear", gamma = 2)
            if len(self._L) > 0: self.learn()

        #@overrides(RecurrentClassifier.Policy)
        def action(self, f, y_star = None):
            return self._classifier.predict([f])[0]

        #@overrides(RecurrentClassifier.Policy)
        def score(self, L = None):
            if self._L is None: raise Exception("Policy's classifier untrained")
            if L is None: L = self._L
            X, Y = self.decomp()
            return self._classifier.score(X, Y)

        def learn(self, L = None):
            """ Learn classifier based on classification set L """
            # (Possibly redundantly) update example set
            if L is not None: self._L = L
            else: L = self._L
            X, Y = self.decomp()
            self._classifier.fit(X, Y)

    class OraclePolicy(Policy):

        __name__ = "OraclePolicy"

        def __init__(self, L): super().__init__(L)

        #@overrides(RecurrentClassifier.Policy)
        def action(self, f, y_star = None):
            # Overrides finding the label on our own
            if y_star is not None: return y_star
            for lf, label in self._L:
                if np.array_equal(f, lf): return label
            return None

        #@overrides(RecurrentClassifier.Policy)
        def score(self, L = None): return 1

    @overrides(Model)
    def train(self, D, *args):
        raise NotImplementedError("Class method 'train' has not been implemented")

    @overrides(Model)
    def test(self, D):

        print("Testing classifier...", flush = True)
        recurrent_hamming_loss = 0
        oracle_hamming_loss = 0
        for x, y in D[:]:

            # Construct: Generated structured output, with each predicted label
            # at time step t influencing all steps from t + 1 to T
            # Oracle: Ideal structured output at each time step t, allowing each
            # successing prediction to be I.I.D.
            y_partial_construct = []
            y_partial_oracle = []

            T = len(y)
            for t in range(T):

                y_partial_optimal = y[:t]
                if t == 0:
                    # Handle dummy label condition, when y_partial_current would be
                    # empty (i.e. []) if not dealt with
                    y_partial_construct = [self._dummy_label]
                    y_partial_oracle = [self._dummy_label]
                if t == 1:
                    # Remove dummy label
                    y_partial_construct = y_partial_construct[1:]
                    y_partial_oracle = y_partial_oracle[1:]

                # Partial output features (for recurrent and oracle)
                f_construct = self._phi(x, y_partial_construct)
                f_optimal = self._phi(x, y_partial_optimal)

                # Predict a label given features generated from both optimal
                # and constructed (i.e. potentially non-optimal) outputs
                y_hat_construct = self._policy.action(f_construct)
                y_hat_oracle = self._policy.action(f_optimal)

                # Build-up outputs based on predictions
                y_partial_construct.append(y_hat_construct)
                y_partial_oracle.append(y_hat_oracle)

            # Compare recurrent and oracle Hamming accuracy
            recurrent_num_wrong_labels = list_diff(y, y_partial_construct)
            oracle_num_wrong_labels = list_diff(y, y_partial_oracle)
            recurrent_hamming_loss += recurrent_num_wrong_labels / len(y)
            oracle_hamming_loss += oracle_num_wrong_labels / len(y)

            # Present errors
            #print(give_err_bars(self._alphabet, y, y_partial_construct), flush = True)

        # Compute and report accuracies
        recurrent_hamming_accuracy = (1.0 - (recurrent_hamming_loss / len(D)))
        oracle_hamming_accuracy = (1.0 - (oracle_hamming_loss / len(D)))
        print("Done: Oracle (Hamming) accuracy is {}%".format(round(oracle_hamming_accuracy * 100)))
        print("Done: Recurrent (Hamming) accuracy is {}%\n".format(round(recurrent_hamming_accuracy * 100)))

        return recurrent_hamming_accuracy

    # NOTE: Single underscore is used to prevent "name mangling", which will now
    # allow inhereting classes to use this method as their own
    def _generate_examples(self, D):
        """ Populate list of classification examples given by "expert" """

        seen_set = set()

        # Through each training example
        for x, y in D[:]:

            # TODO: See if this helps overfitting
            #if "".join(y) in seen_set: continue
            seen_set.add("".join(y))

            # Add training example for each left-anchored subset of the
            # structured output y; (left-anchored subsets: "h", "he", "hey")
            T = len(y)
            for t in range(T):

                y_t = y[t]
                y_partial = y[:t]
                if t == 0: y_partial = [self._dummy_label]

                #print("".join(y_partial), y_t, flush = True)

                # Generate features, package up, and append to list
                f = self._phi(x, y_partial)
                example = (f, y_t)
                self._L.append(example)

        # TODO: See if shuffling helps (done in other places too)
        random.shuffle(self._L)

class ImitationClassifier(RecurrentClassifier):

    __name__ = "ImitationRecurrentClassifier"

    def __init__(self, alphabet, len_x):
        super().__init__(alphabet, len_x)

    @overrides(RecurrentClassifier)
    def train(self, D, *args):

        # TODO: Investigate OCR overfitting

        print("Generating examples...", flush = True)
        self._generate_examples(D)

        # NOTE: Attempted custom, point-by-point training with the library
        # classifier; but, this gave worse accuracy

        # Train SVM on our generated examples
        print("Training classifier...", flush = True)
        self._policy.learn(self._L)

        # Training accuracy
        oracle_hamming_accuracy = self._policy.score()
        print("Done: Oracle (Hamming) accuracy is {}%\n".format( \
            round(oracle_hamming_accuracy * 100)))
        return oracle_hamming_accuracy

class DAggerClassifier(RecurrentClassifier):
    """
    DAgger: Dataset Aggregation (Recurrent Classifier)
    This classifier is within the Follow-the-Leader set of classifiers

    The intuition behind this algorithm is to aggregate data which our learned
    policies will be more likely to see, based on past experience; this means
    training on trajectories that are not optimal, but also providing a direction
    for them to go to recover
    """

    # TODO: Address issues with not seeing improvement with the DAgger algorithm

    __name__ = "DAggerRecurrentClassifier"

    def __init__(self, alphabet, len_x, beta = 0.5):
        super().__init__(alphabet, len_x)
        self._default_it = 5    # Default number of DAgger iterations
        self._beta = beta       # Interpolation parameter
        self._policy = None

    @overrides(RecurrentClassifier)
    def train(self, D, *args):

        # Determine DAgger iteration count
        if len(args) > 0: d_max = args[0]
        else: d_max = self._default_it

        # Splitup training and validation data, 75/25; shuffle D to ensure
        # uniform distribution; essentially, we are doing 4-fold cross
        # validation (as our resampling technique), except we are resampling
        # across each run of the program, not within the 'train' method itself
        random.shuffle(D)
        v_split = int(0.75 * len(D))
        train_data = D[:v_split]
        validation_data = D[v_split:]

        # As if oracle classifier is giving us trajectories
        print("Generating examples...", flush = True)
        self._generate_examples(train_data)

        # Train SVM on our generated examples
        print("Training classifier...", flush = True)

        # Instantiate initial classifier
        h_learned = self.LearnedPolicy(self._L)
        h_oracle = self.OraclePolicy(self._L)
        h_history = [] # History of learned classifiers through iterations
        beta_const = self._beta

        # Data aggregation iterations
        for j in range(d_max):

            # Exponential decay of beta over iterations, meaning less and less
            # reliance on the oracle classifier!
            self._beta = math.pow(beta_const, j + 1)
            h_current = self._choose_policy(h_oracle, h_learned)
            print("\t", j + 1, d_max, h_current.__name__, len(h_current._L), \
                "beta = {}".format(self._beta), flush = True)

            for x, y in train_data[:]:

                # Structured output we're going to be building up; note that
                # this is considered the current policy's trajectory
                y_partial_construct = []

                # Iterated select partial outputs of y
                T = len(y)
                for t in range(T):

                    # Optimal label
                    y_star = y[t]

                    # Generate features
                    if t == 0: y_partial_construct = [self._dummy_label]
                    if t == 1: y_partial_construct = y_partial_construct[1:]
                    f = self._phi(x, y_partial_construct)

                    # Does current classifier match oracle classification?
                    # NOTE: Having an oracle policy here isn't doing much
                    c_action = h_current.action(f, y_star)  # Predicted action
                    o_action = h_oracle.action(f, y_star)   # Oracle action
                    # TODO: Uncomment the below Boolean expression, as this
                    # is currently adding EVERY
                    #if True: #c_action == o_action:
                    # NOTE: Modified original condition for adding
                    if False: #c_action == o_action:

                        # Add classification example (from oracle)
                        clf_example = (f, o_action)
                        self._L.append(clf_example)

                    # Add to policy trajectory
                    y_partial_construct.append(c_action)

            # Store history of all learned policies, as well as Learn a new
            # classifier from aggregate data! Also, only add policy to history
            # if it's not an oracle policy
            if not isinstance(h_current, self.OraclePolicy):
                h_history.append(h_learned)
            h_learned = self.LearnedPolicy(self._L)

        # Find best learned classifier based on validation data
        print("Determining best learned policy to pick...")
        h_best_tup = (None, -1)
        disable_stdout()
        for h in h_history:
            self._policy = h
            acc = self.test(validation_data)
            if acc > h_best_tup[1]: h_best_tup = (h, acc)
        enable_stdout()

        # Set our policy to the max learned classifier
        self._policy = h_best_tup[0]

        # Training accuracy
        oracle_hamming_accuracy = self._policy.score()
        print("Done: Oracle (Hamming) accuracy is {}%\n".format( \
            round(oracle_hamming_accuracy * 100)))
        return oracle_hamming_accuracy

    def _choose_policy(self, a, b):
        """
        Choose policy a with probability beta, and policy b with probability
        (1 - beta)
        """
        choice_point = random.random()
        if choice_point < self._beta: return a
        return b
