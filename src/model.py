# model.py

"""
MODEL
Scaffold for structured prediction classifiers
"""

class Model:
    __name__ = "Model"

    def __init__(self, alphabet, len_x):

        # Structure-related
        self._alphabet = alphabet
        self._len_x = len_x
        self._len_y = len(alphabet)
        self._convergence_range = 1 # Considering accuracy range [0, 100]

    def train(self, D, *args): raise NotImplementedError()
    def test(self, D): raise NotImplementedError()

    def _converged(self, diff):
        """
        Given difference between two numbers, determine if they are
        within the convergence range
        """
        diff = int(diff * 100)
        if diff <= self._convergence_range: return True
        return False
