# model.py

import pyqtgraph as pg
import numpy as np
import random
import time
import copy

from util import *

"""
STRUCTURED PERCEPTRON
Methods immediately relevant to the concept of a generalized perceptron
"""

class StructuredPerceptron:

    # NOTE: Any variable declared outside of the constructor is a static
    # variable, and will change all instances of this class if they are changed;
    # however, object variables (declared in __init__) are unique to the object.

    def __init__(self, alphabet, len_x, phi_order, R, eta, MAX, b = None):

        self.alphabet = alphabet
        self.len_x = len_x
        self.len_y = len(alphabet)

        # Detect and set phi properties dynamically
        self.phi_order = phi_order
        self.phi_funcs = [self.phi_unary, \
                 self.phi_pairwise, \
                 self.phi_third_order, \
                 self.phi_fourth_order]
        phi = self.phi_funcs[self.phi_order - 1]

        # Perceptron-related
        self.R = R                          # Number of restarts
        self.eta = eta                      # Learning rate
        self.MAX = MAX                      # Maximum number of iterations
        self.w = None                       # Learned weight vector

        # Phi-related
        self.phi = phi                      # Joint-feature function
        self.phi_dimen = -1
        self.pairwise_base_index = -1
        self.triplet_base_index = -1
        self.quadruplet_base_index = -1
        self.pairs = []
        self.triplets = []
        self.quadruplets = []

        # (Optional) beam width
        self.b = b

    def train(self, D):
        """ Train on input data set D """

        # Display heading
        self.display_header(D)

        # Record model's progress w.r.t. accuracy (and iteration improvment)
        it_improvement = np.zeros((len(D)))
        acc_progress = []
        pw = pg.plot()

        # Reset weights of scoring function to 0
        if self.w is not None: self.w.fill(0)

        # Iterate until max iterations or convergence
        for it in range(self.MAX):

            # Time each iteration
            it_start = time.clock()

            print("[Iteration " + str(it) + "]\n")

            # Essential iteration-related vars
            train_num = 0
            num_mistakes = 0
            num_correct = 0

            # Go through training examples
            for x, y in D[:]:

                # Skip empty data points
                # TODO: See if we can remove this
                if len(x) < 1: continue

                # Predict, i.e. run inference
                y_hat = self.bstfbs(x, len(y)) #self.rgs(x, len(y))
                num_right_chars = len(y) - list_diff(y_hat, y)

                # TODO: Reformat
                instance_str = ("Train instance " + str(it) \
                    + "." + str(train_num))

                # If error, update weights
                if y_hat != y:

                    instance_str = ("\t[-]\t" + instance_str + "\t("
                                    + str(num_right_chars) + "/" + str(len(y)) + ")")

                    # Compute phi's
                    ideal_phi = self.phi(x, y)
                    pred_phi = self.phi(x, y_hat)

                    # Perform weight update
                    self.w = np.add(self.w, np.dot(self.eta, \
                        (np.subtract(ideal_phi, pred_phi))))

                    num_mistakes += 1

                else:
                    instance_str = ("\t[+]\t" + instance_str + "\t(" + str(len(y))
                                    + "/" + str(len(y)) + ")")
                    num_correct += 1

                instance_str += ("\t[" + str(num_correct) + "/"
                                 + str(train_num + 1) + "]")

                # Measure iteration improvement (compared to last)
                if (it > 0):
                    improvement = num_right_chars - it_improvement[train_num]
                    if improvement != 0:
                        instance_str += "\t" + give_sign(int(improvement))
                it_improvement[train_num] = num_right_chars

                # Print instance details and update training number
                print(instance_str)
                train_num += 1

            # Determine accuracy
            accuracy = num_correct / (num_correct + num_mistakes)
            acc_progress.append(accuracy)

            # Plot accuracy timeline
            if len(acc_progress) > 1:
                pw.plot(acc_progress, clear = True)
                pg.QtGui.QApplication.processEvents()

            # Report iteration stats
            print()
            print("\t| Accuracy = " + str(accuracy * 100) + "%")
            print("\t| Number correct = " + str(num_correct))
            print("\t| Number of mistakes = " + str(num_mistakes))
            print("\t| Time = ~" + str(round((time.clock() - it_start) / 60))
                  + " min")
            print()

            # Check for convergence
            if len(acc_progress) > 1:
                if acc_progress[-1] == acc_progress[-2]:
                    print("Model has converged:")
                    print("\tAccuracy = " + str(accuracy * 100) + "%")
                    print("\tIteration = " + str(it))
                    break

        # Return and save weights!
        return

    def test(self):
        """
        Train model with our outside-loaded or trained weights, return accuracy
        """

        # TODO

        return 0.0

    def bstfbs(self, x, len_y):
        """
        Best-First Beam Search inference
        """

        # TODO: Modularize for both early update and max-violation
        h = lambda y: self.get_score(x, y)
        bs = BeamSearch(self.b, h, len_y, self.alphabet)
        y_hat = bs.search()

        return y_hat

    def bdthfbs(self, x, len_y):
        """
        Breadth-First Beam Search inference
        """

        # TODO

        pass

    def rgs(self, x, len_y):
        """
        Randomized Greedy Search (RGS) inference:
        Try and use the current weights to arrive at the correct label;
        we will always return our best guess
        """

        for i in range(self.R):

            # Initialize best scoring output randomly
            y_hat = self.get_random_y(len_y)

            # Until convergence
            while True:

                # Get max char
                y_max = self.get_max_one_char(x, y_hat)
                if y_max == y_hat: break
                y_hat = y_max

        return y_hat

    def phi_unary(self, x, y):
        """
        Unary joint-feature function: sums together feature vectors of
        similar labels, and combines features for different labels in different
        indices
        """

        dimen = self.len_x * self.len_y
        vect = np.zeros((dimen))

        # NOTE: Changed from len(x) to len(y) to account for 
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

    def phi_pairwise(self, x, y):
        """
        Pairwise joint-feature function:
            0. Do unary features
            1. Capture all two-char permutations
            2. Assign these permuations consistent indices
            3. Count frequencies of each permuation and update vector
               at that index
        """

        global pairwise_base_index, pairs

        # Initial setting of phi dimensions
        pairwise_base_index = self.len_x * self.len_y
        dimen = (self.len_x * self.len_y) + (self.len_y ** 2)

        vect = np.zeros((dimen))
        alpha_list = list(self.alphabet)
        alpha_list.sort()

        # (One-time) Generate pair-index object
        if len(pairs) == 0:
            for a in alpha_list:
                for b in alpha_list:
                    p = a + b
                    pairs.append(p)

        # Unary features
        for i in range(len(x)):

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
            comb_index = pairs.index(p)
            vect_index = pairwise_base_index + comb_index

            # Update occurace of pair
            vect[vect_index] += 1

        return vect

    def phi_third_order(self, x, y):
        """
        Third-order joint-feature function:
            0. Do unary features
            1. Do pairwise features
            2. Capture all three-char permutations
            3. Assign these permuations consistent indices
            4. Count frequencies of each permuation and update vector
               at that index
        """

        global pairwise_base_index, triplet_base_index, pairs, triplets

        # Initial setting of phi dimensions
        pairwise_base_index = self.len_x * self.len_y
        triplet_base_index = pairwise_base_index + (self.len_y ** 2)
        dimen = triplet_base_index + (self.len_y ** 3)

        vect = np.zeros((dimen))
        alpha_list = list(self.alphabet)
        alpha_list.sort()

        # (One-time) Generate pair and triplet lists
        if len(triplets) == 0:
            for a in alpha_list:
                for b in alpha_list:
                    # Grab pair
                    p = a + b
                    pairs.append(p)

                    for c in alpha_list:
                        # Grab triplet
                        t = a + b + c
                        triplets.append(t)

        # Unary features
        for i in range(len(x)):

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
            comb_index = pairs.index(p)
            vect_index = pairwise_base_index + comb_index

            # Update occurace of pair
            vect[vect_index] += 1

        # Third-order features
        for i in range(len(y) - 2):

            # Get pair index
            a = y[i]
            b = y[i + 1]
            c = y[i + 2]
            t = a + b + c
            comb_index = triplets.index(t)
            vect_index = triplet_base_index + comb_index

            # Update occurace of triplet
            vect[vect_index] += 1

        return vect

    def phi_fourth_order(self, x, y):
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

        global pairwise_base_index, triplet_base_index, quadruplet_base_index
        global pairs, triplets

        # Initial setting of phi dimensions
        pairwise_base_index = self.len_x * self.len_y
        triplet_base_index = pairwise_base_index + (self.len_y ** 2)
        quadruplet_base_index = triplet_base_index + (self.len_y ** 3)
        dimen = quadruplet_base_index + (self.len_y ** 4)

        vect = np.zeros((dimen))
        alpha_list = list(self.alphabet)
        alpha_list.sort()

        # (One-time) Generate pair, triplet, and quadruplet lists
        if len(quadruplets) == 0:
            for a in alpha_list:
                for b in alpha_list:
                    # Grab pair
                    p = a + b
                    pairs.append(p)

                    for c in alpha_list:
                        # Grab triplet
                        t = a + b + c
                        triplets.append(t)

                        for d in alpha_list:
                            # Grab quadruplet
                            q = a + b + c + d
                            quadruplets.append(q)

        # Unary features
        for i in range(len(x)):

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
            comb_index = pairs.index(p)
            vect_index = pairwise_base_index + comb_index

            # Update occurace of pair
            #print(p, "occurs at", vect_index)
            vect[vect_index] += 1

        # Third-order features
        for i in range(len(y) - 2):

            # Get pair index
            a = y[i]
            b = y[i + 1]
            c = y[i + 2]
            t = a + b + c
            comb_index = triplets.index(t)
            vect_index = triplet_base_index + comb_index

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
            comb_index = quadruplets.index(q)
            vect_index = quadruplet_base_index + comb_index

            # Update occurace of quadruplet
            vect[vect_index] += 1

        return vect

    def get_score(self, x, y_hat):
        """
        Compute score of joint-feature function with weights,
        while also setting the weight dimensions to phi dimensions
        dynamically
        """

        # Joint-feature function of predicted label-group for input x
        pred_phi = self.phi(x, y_hat)

        print("Pred_phi", pred_phi)

        # If weights are unset, dynamically init (based on phi length)
        if self.w is None: self.w = np.zeros((len(pred_phi)))

        return np.dot(self.w, pred_phi)

    def get_random_y(self, len_y):
        """ Return random array of alphabet characters of given length """

        rand_y = []

        # If no length passed
        if len_y == None:
            min_word_len = 2
            max_word_len = 6
            rand_word_len = math.floor(random.uniform(min_word_len,
                                                      max_word_len + 1))
        else: rand_word_len = len_y

        for i in range(rand_word_len):
            rand_char = random.sample(self.alphabet, 1)[0]
            rand_y.append(rand_char)

        return rand_y

    def get_max_one_char(self, x, y_hat):
        """
        Make one-character changes to y_hat, finding which
        single change produces the best score; we return
        that resultant y_max
        """

        # Initialize variables to max
        s_max = self.get_score(x, y_hat)
        y_max = y_hat

        for i in range(len(y_hat)):

            # Copy of y_hat to play with
            y_temp = copy.deepcopy(y_hat)

            # Go through a-z at i-th index
            for c in self.alphabet:

                # Get score of 1-char change
                y_temp[i] = c
                s_new = self.get_score(x, y_temp)

                # Capture highest-scoring change
                if s_new > s_max:
                    s_max = s_new
                    y_max = y_temp

        return y_max

    def set_weights(self, w):
        """ Allow the outside to set our scoring function weights """

        self.w = w

    def display_header(self, D):
        print()
        print("Structured Perceptron:")
        print()
        print("\tData length (with limitation) = " + str(len(D)))
        print("\tNumber of restarts = " + str(self.R))
        print("\tLearning rate = " + str(self.eta))
        print("\tMax iteration count = " + str(self.MAX))
        print("\tOrder of joint-features = " + str(self.phi_order))
        print("\tAlphabet length = " + str(self.len_y))
        print()

class BeamSearch:

    def __init__(self, b, h, term_len, alphabet):
        """ Construct beam properties necessary for heuristic search """

        # Beam width
        self.b = b

        # Heuristic guiding beam search; tells us desirability of inputted
        # node given it's structured data (as a list)
        self.h = h

        # Length of terminal output
        self.term_len = term_len

        # Alphabet used for constructing nodes in the search space
        self.alphabet = alphabet

    def search(self):
        """
        Move through search space guided by given heuristic h, stopping
        search once one node in beam is of given terminal length
        """

        beam = []
        candidates = []
        y_select = None

        # Loop until complete structure output found in beam
        while True:

            # Maximum scoring output in beam
            y_select = self.max_in_beam(beam)

            # Expand upon maximum scoring (partial) output
            candidates = beam + self.gen_children(y_select)

            # Prune excess, lower-scoring nodes
            beam = sorted(candidates, key = lambda y: self.h(y))[:b]

            # Check for terminal node/output
            for out in beam:
                if len(out) == self.term_len: break

        # Get highest scoring, complete output in beam (and return)
        beam = [y for y in beam if len(y) == self.term_len]
        y_hat = self.max_in_beam(beam)

        return y_hat

    def max_in_beam(self, beam):
        """ Return maximum scoring node in beam w.r.t. heuristic h """

        beam_node_scores = {(y, self.h(y)) for y in beam}
        return max(beam_node_scores, key = (lambda y: beam_node_scores[y]), \
                   default = [])

    def gen_children(self, y):
        """
        Using alphabet and given (partial) labelling y, generate list of
        children, each with 1-char difference from parent
        """

        children = []
        for char in self.alphabet:
            child = y + [char]
            children.append(child)
        return children
