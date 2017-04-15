# model.py

"""
MODEL
Scaffold for structured prediction classifiers
"""

class Model:
    __name__ = "Model"

    def __init__(self, alphabet, len_x, phi_order):

        # Structure-related
        self._alphabet = alphabet
        self._len_x = len_x
        self._len_y = len(alphabet)
        self._convergence_range = 1 # Considering accuracy range [0, 100]

        # Phi: joint-feature function
        self._phi_order = phi_order
        self._first_phi_run = False

    def train(self, D, *args): raise NotImplementedError()
    def test(self, D): raise NotImplementedError()

    def _gen_group_dict():
        """ Generate all group permutations """
        # Group-to-index dictionary doesn't matter for unary features
        if self._phi_order == 0: return {}
        group_to_index = {}
        for o in range(1, self._phi_order):
            group_len = o + 1
            self._gen_group_perms(glen = group_len, gdict = group_to_index)
        return group_to_index

    def _gen_group_perms(self, gdict, glen, prefix = ""):
        """
        Recursively generate every permutation of group's lenth with our alphabet
        """
        if len(prefix) == glen:
            gdict[prefix] = len(gdict)
        for a in self._alpha_list:
            self._gen_group_perms(gdict = gdict, glen = glen, prefix = (prefix + a))

    @property
    # TODO: --------------------------
    def _phi_dimen(self):
        res = (self._len_x * self._len_y) + (self._len_y ** 2)
        return 0

    # TODO: Generalize _phi to dynamically create itself from just the _phi_order
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
        if self._first_phi_run:

            self._first_phi_run = False

            # Keep alphabet order deterministic
            self._alpha_list = list(self._alphabet)
            self._alpha_list.sort()

            # Initial setting of phi dimensions
            self._base_indices = []
            self._group_to_index = self._gen_group_dict()

        # Collect features within phi's order
        features = np.zeros((self._dimen))

        # TODO: Handle all past-unary features dynamically

        # Unary features
        for i in range(len(y)):
            index = self._alpha_list.index(y[i])
            x_vect = np.array(x[i])
            y_target = len(x[i]) * index
            for j in range(len(x[i])): features[j + y_target] += x_vect[j]

        # Pairwise features
        for i in range(len(y) - 1):

            # TODO: replace with _phi_dimen?
            self._phi_pairwise_base_index = self._len_x * self._len_y

            # Get pair index
            a = y[i]
            b = y[i + 1]
            p = a + b
            comb_index = self._phi_pairs.index(p)
            vect_index = self._phi_pairwise_base_index + comb_index

            # Update occurace of pair
            features[vect_index] += 1

        return features

    # Phi methods

    def __phi_unary(self, x, y):
        """
        Unary joint-feature function: sums together feature vectors of
        similar labels, and combines features for different labels in different
        indices
        """

        dimen = self.len_x * self.len_y
        vect = np.zeros((dimen))

        # NOTE: Changed from len(x) to len(y) to account for partial outputs
        # of y; now we skill unlabelled x_i (character) vectors within the
        # larger x (word) vector
        for i in range(len(y)):

            x_i = x[i]
            y_i = y[i]

            # Sorting keeps consistency of indices with respect to
            # all prior and following phi(x, y) vectors
            alpha_list = list(self.alphabet)
            alpha_list.sort()
            index = alpha_list.index(y_i)
            x_vect = np.array(x_i)

            # Manual insertion of x into standard vector
            # NOTE: Holy fuck, had "= x_vect[j]" before, not "+="
            y_target = len(x_i) * index
            for j in range(len(x_i)): vect[j + y_target] += x_vect[j]

        return vect

    def __phi_pairwise(self, x, y):
        """
        Pairwise joint-feature function:
            0. Do unary features
            1. Capture all two-char permutations
            2. Assign these permuations consistent indices
            3. Count frequencies of each permuation and update vector
               at that index
        """

        # Initial setting of phi dimensions
        self.__pairwise_base_index = self.len_x * self.len_y
        dimen = (self.len_x * self.len_y) + (self.len_y ** 2)

        vect = np.zeros((dimen))
        alpha_list = list(self.alphabet)
        alpha_list.sort()

        # (One-time) Generate pair-index object
        if len(self.__pairs) == 0:
            for a in alpha_list:
                for b in alpha_list:
                    p = a + b
                    self.__pairs.append(p)

        # Unary features
        for i in range(len(y)):

            x_i = x[i]
            y_i = y[i]

            # Unary features
            index = alpha_list.index(y_i)
            x_vect = np.array(x_i)
            y_target = len(x_i) * index
            for j in range(len(x_i)): vect[j + y_target] += x_vect[j]

        # Pairwise features
        for i in range(len(y) - 1):

            # Get pair index
            a = y[i]
            b = y[i + 1]
            p = a + b
            comb_index = self.__pairs.index(p)
            vect_index = self.__pairwise_base_index + comb_index

            # Update occurace of pair
            vect[vect_index] += 1

        return vect

    def __phi_third_order(self, x, y):
        """
        Third-order joint-feature function:
            0. Do unary features
            1. Do pairwise features
            2. Capture all three-char permutations
            3. Assign these permuations consistent indices
            4. Count frequencies of each permuation and update vector
               at that index
        """

        # Initial setting of phi dimensions
        self.__pairwise_base_index = self.len_x * self.len_y
        self.__triplet_base_index = self.__pairwise_base_index + (self.len_y ** 2)
        dimen = self.__triplet_base_index + (self.len_y ** 3)

        vect = np.zeros((dimen))
        alpha_list = list(self.alphabet)
        alpha_list.sort()

        # (One-time) Generate pair and triplet lists
        if len(self.__triplets) == 0:
            for a in alpha_list:
                for b in alpha_list:
                    # Grab pair
                    p = a + b
                    self.__pairs.append(p)

                    for c in alpha_list:
                        # Grab triplet
                        t = a + b + c
                        self.__triplets.append(t)

        # Unary features
        for i in range(len(y)):

            x_i = x[i]
            y_i = y[i]

            # Unary features
            index = alpha_list.index(y_i)
            x_vect = np.array(x_i)
            y_target = len(x_i) * index
            for j in range(len(x_i)): vect[j + y_target] += x_vect[j]

        # Pairwise features
        for i in range(len(y) - 1):

            # Get pair index
            a = y[i]
            b = y[i + 1]
            p = a + b
            comb_index = self.__pairs.index(p)
            vect_index = self.__pairwise_base_index + comb_index

            # Update occurace of pair
            vect[vect_index] += 1

        # Third-order features
        for i in range(len(y) - 2):

            # Get pair index
            a = y[i]
            b = y[i + 1]
            c = y[i + 2]
            t = a + b + c
            comb_index = self.__triplets.index(t)
            vect_index = self.__triplet_base_index + comb_index

            # Update occurace of triplet
            vect[vect_index] += 1

        return vect

    def __phi_fourth_order(self, x, y):
        """
        Fourth-order joint-feature function:
            0. Do unary features
            1. Do pairwise features
            2. Do third-order features
            3. Capture all four-char permutations
            4. Assign these permuations consistent indices
            5. Count frequencies of each permuation and update vector
               at that index
        """

        # Initial setting of phi dimensions
        self.__pairwise_base_index = self.len_x * self.len_y
        self.__triplet_base_index = self.__pairwise_base_index + (self.len_y ** 2)
        self.__quadruplet_base_index = self.__triplet_base_index + (self.len_y ** 3)
        dimen = self.__quadruplet_base_index + (self.len_y ** 4)

        vect = np.zeros((dimen))
        alpha_list = list(self.alphabet)
        alpha_list.sort()

        # (One-time) Generate pair, triplet, and quadruplet lists
        if len(self.__quadruplets) == 0:
            for a in alpha_list:
                for b in alpha_list:
                    # Grab pair
                    p = a + b
                    self.__pairs.append(p)

                    for c in alpha_list:
                        # Grab triplet
                        t = a + b + c
                        self.__triplets.append(t)

                        for d in alpha_list:
                            # Grab quadruplet
                            q = a + b + c + d
                            self.__quadruplets.append(q)

        # Unary features
        for i in range(len(y)):

            x_i = x[i]
            y_i = y[i]

            # Unary features
            index = alpha_list.index(y_i)
            x_vect = np.array(x_i)
            y_target = len(x_i) * index
            for j in range(len(x_i)): vect[j + y_target] += x_vect[j]

        # Pairwise features
        for i in range(len(y) - 1):

            # Get pair index
            a = y[i]
            b = y[i + 1]
            p = a + b
            comb_index = self.__pairs.index(p)
            vect_index = self.__pairwise_base_index + comb_index

            # Update occurace of pair
            vect[vect_index] += 1

        # Third-order features
        for i in range(len(y) - 2):

            # Get pair index
            a = y[i]
            b = y[i + 1]
            c = y[i + 2]
            t = a + b + c
            comb_index = self.__triplets.index(t)
            vect_index = self.__triplet_base_index + comb_index

            # Update occurace of triplet
            vect[vect_index] += 1

        # Fourth-order features
        for i in range(len(y) - 3):

            # Get pair index
            a = y[i]
            b = y[i + 1]
            c = y[i + 2]
            d = y[i + 3]
            q = a + b + c + d
            comb_index = self.__quadruplets.index(q)
            vect_index = self.__quadruplet_base_index + comb_index

            # Update occurace of quadruplet
            vect[vect_index] += 1

        return vect

    def _converged(self, diff):
        """
        Given difference between two numbers, determine if they are
        within the convergence range
        """
        diff = int(diff * 100)
        if diff <= self._convergence_range: return True
        return False
