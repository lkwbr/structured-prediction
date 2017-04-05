# recurrent.py

"""
RECURRENT CLASSIFIER
xxx
"""

import numpy as np

from util import *
from model import Model

class RecurrentClassifier(Model):
    """
    Reduction of structured prediction to multi-label classification
    """

    __name__ = "RecurrentClassifier"

    def __init__(self, alphabet, len_x):

        # Have parent do their thing
        super().__init__(alphabet, len_x)

        # L: Set of classification examples
        self.L = []

    @overrides(Model)
    def train(self, D):

        self.__generate_examples(D)

        return 0.0

    @overrides(Model)
    def test(self, D):
        return 0.0

    def __generate_examples(self, D):
        """ Populate list of classification examples """

        # Through each training example
        for x, y in D[:]:

            # NOTE: Use something as a dummy label

            # Add training example for each left-anchored subset of the
            # structured output y; (left-anchored subsets: "h", "he", "hey")
            T = len(y)
            for t in range(T):

                y_partial = y[:t]
                y_t = y[t]

                # Generate features, package up, and append to list
                f = self.__generate_features(x, y_partial)
                example = (f, y_t)
                self.L.append(example)

    def __generate_features(self, x, y):
        """  """

        # TODO: Check if this is what Jana was suggesting

        f = np.zeros((self.len_x))
        for i in len(y):
            f = np.add(f, x[i])

        return f
