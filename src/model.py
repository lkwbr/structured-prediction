# model.py

"""
MODEL
Scaffold for structured prediction classifiers
"""

import numpy as np
import copy

class Model:
    __name__ = "Model"

    def __init__(self, alphabet, len_x, phi_order):

        # Structure-related
        self._alphabet = alphabet
        self._len_x = len_x
        self._len_y = len(alphabet)
        self._convergence_range = 1 # Considering accuracy range [0, 100]

        # Label stuff
        self._label_len = len(min(alphabet, key = len))
        self._dummy_label = "$" * self._label_len
        alphabet.add(self._dummy_label)

        # Phi: joint-feature function
        self._phi_order = phi_order
        self._first_phi_run = True

    def train(self, D, *args): raise NotImplementedError()
    def test(self, D): raise NotImplementedError()

    def _setup_phi(self):
        """ Generate all group permutations """

        # Keep alphabet order deterministic
        self._alpha_list = list(self._alphabet)
        self._alpha_list.sort()

        # Initial setting of phi dimensions
        # Base index = end of unary features, start of grouped features
        self._group_base_index = self._len_x * self._len_y
        self._group_to_index = {}

        # Generate group dictionary
        for o in range(1, self._phi_order + 1):
            group_len = o + 1
            self._gen_group_perms(glen = group_len)

        # Set phi dimensionality
        self._phi_dimen = self._group_base_index + len(self._group_to_index)

        # Don't run again!
        self._first_phi_run = False

    def _gen_group_perms(self, glen, prefix = ""):
        """
        Recursively generate every permutation of group's lenth with our alphabet
        """

        # Terminating condition, generated a permutation/prefix
        if len(prefix) / self._label_len == glen:
            uniq_index = self._group_base_index + len(self._group_to_index)
            self._group_to_index[prefix] = uniq_index
            return

        for a in self._alpha_list:
            new_prefix = prefix + a
            self._gen_group_perms(
                glen = glen,
                prefix = new_prefix)

    def _get_perms(self, item, n):
        """ Get all n-ary permutations of given item """

        perms = []
        for i in range(len(item) - (n - 1)):
            p = "".join(item[i:(i + n)])
            perms.append(p)
        return perms

    def _phi(self, x, y):
        """
        Dynamic joint-feature function, where phi order implies the following:
            0 -> unary features
            1 -> unary + pairwise features
            2 -> unary + pairwise + trinary features
            etc.
        Supports generating features for partially labelled outputs, i.e.,
        for situations where len(y) <= len(x)
        """

        if self._phi_order < 0: raise ValueError("Order of phi must be >= 0")

        # (One-time) Do initial optimizations
        if self._first_phi_run: self._setup_phi()

        # Collect features within phi's order
        features = np.zeros((self._phi_dimen))

        # Unary features
        for i in range(len(y)):
            index = self._alpha_list.index(y[i])
            x_vect = np.array(x[i])
            y_target = len(x[i]) * index
            for j in range(len(x[i])): features[j + y_target] += x_vect[j]

        # Post-unary features
        for o in range(1, self._phi_order + 1):

            # Get all permuations of length o
            group_len = o + 1
            perms = self._get_perms(y, group_len)
            for p in perms:
                # Update occurace of single grouping
                vect_index = self._group_to_index[p]
                features[vect_index] += 1

        return features

    def _converged(self, diff):
        """
        Given difference between two numbers, determine if they are
        within the convergence range
        """
        diff = int(diff * 100)
        if diff <= self._convergence_range: return True
        return False
