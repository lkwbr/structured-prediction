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

    def __init__(self, alphabet, len_x, phi_order, update_method, search_type, \
        R, eta, MAX, b):

        # Candy shop - so many choices!
        self.phi_funcs = [ \
            self.phi_unary, \
            self.phi_pairwise, \
            self.phi_third_order, \
            self.phi_fourth_order]
        self.update_methods = [ \
            self.standard_update, \
            self.early_update, \
            self.max_violation_update]
        self.search_types = [ \
            BestFirstBeamSearch,
            BreadthFirstBeamSearch]

        # Structure-related
        self.alphabet = alphabet
        self.len_x = len_x
        self.len_y = len(alphabet)

        # Detect and set phi properties dynamically
        self.phi_order = phi_order
        phi = self.phi_funcs[self.phi_order - 1]

        # Perceptron-related
        self.R = R                          # Number of restarts
        self.eta = eta                      # Learning rate
        self.MAX = MAX                      # Maximum number of iterations
        self.w = None                       # Learned weight vector
        self.update_method = self.update_methods[update_method]
        self.convergence_range = 1          # When considering [0, 100]

        # Phi-related
        self.phi = phi                      # Joint-feature function
        self.phi_dimen = -1                 # Dimensionality of phi
        self.pairwise_base_index = -1
        self.triplet_base_index = -1
        self.quadruplet_base_index = -1
        self.pairs = []
        self.triplets = []
        self.quadruplets = []

        # Beam search related
        self.b = b
        self.search_obj = self.search_types[search_type]

    # Main methods: training and testing with provided data

    def train(self, D):
        """ Train on input data set D """

        # Display heading
        self.display_header(D)

        # Record model's progress w.r.t. accuracy (and iteration improvment)
        it_improvement = np.zeros((len(D)))
        acc_progress = []
        ham_acc_progress = []
        #pw = pg.plot()

        # Reset weights of scoring function to 0; useful if we ever train
        # more than once for this same model instance
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
            hamming_loss = 0

            # Go through training examples
            for x, y in D[:]:

                # Skip empty data points
                # TODO: See if we can remove this
                if len(x) < 1: continue

                # Perform standard weight update
                y_hat, correct, mistake, num_right_chars, instance_str, err_display = \
                    self.update_method(x, y, (str(it) + "." + str(train_num)))

                # Compute/determine accuracy stats
                num_correct += correct
                num_mistakes += mistake
                instance_str += ("\t[" + str(num_correct) + "/"
                                 + str(train_num + 1) + "]")
                hamming_loss += list_diff(y_hat, y) / len(y)

                # Measure iteration improvement (compared to last)
                if (it > 0):
                    improvement = num_right_chars - it_improvement[train_num]
                    if improvement != 0:
                        instance_str += "\t" + give_sign(int(improvement))
                it_improvement[train_num] = num_right_chars
                instance_str += err_display

                # Print instance details and update training number
                print(instance_str)
                train_num += 1

            # Determine accuracy
            num_examples = train_num
            accuracy = num_correct / num_examples
            hamming_accuracy = (1.0 - (hamming_loss / num_examples))
            acc_progress.append(accuracy)
            ham_acc_progress.append(hamming_accuracy)

            # Plot accuracy timeline
            last_acc = 0
            last_ham_acc = 0
            if len(ham_acc_progress) > 1:
                # Get accuracy from previous iteration
                last_acc = acc_progress[-2]
                last_ham_acc = ham_acc_progress[-2]
                # NOTE: Uncommented plotting for efficiency
                #pw.plot(acc_progress, clear = True)
                #pg.QtGui.QApplication.processEvents()

            # Accuracy improvement
            acc_it_diff = accuracy - last_acc
            ham_acc_it_diff = hamming_accuracy - last_ham_acc

            # Report iteration stats
            print()
            print("\t| Standard accuracy = " + str(accuracy * 100) + "%" \
                  + "\t" + give_sign(acc_it_diff * 100))
            print("\t| Hamming accuracy = " + str(hamming_accuracy * 100) + "%" \
                  + "\t" + give_sign(ham_acc_it_diff * 100))
            print("\t| Number correct = " + str(num_correct))
            print("\t| Number of mistakes = " + str(num_mistakes))
            print("\t| Time = ~" + str(round((time.clock() - it_start) / 60))
                  + " min")
            print()

            # Check for convergence
            if len(acc_progress) > 1:
                if self.converged(ham_acc_it_diff):
                    print("Model has converged:")
                    print("\tAccuracy = " + str(accuracy * 100) + "%")
                    print("\tIteration = " + str(it))
                    break

        # Return training accuracy
        return hamming_accuracy

    def test(self, D):
        """
        Test model with our model's weights, collect and return two types
        of accuracy:
            - Standard accuracy: (num_totally_correct_outputs / num_outputs);
            - Hamming accuracy: (num_correct_characters / num_total_characters);
        """

        print()
        print("Testing model")
        print()

        train_num = 0
        num_mistakes = 0
        num_correct = 0
        hamming_loss = 0

        # Go through training examples
        for x, y in D[:]:

            # Skip empty data points
            if len(x) < 1: continue

            # Perform standard weight update
            y_hat, correct, mistake, num_right_chars, instance_str = \
                self.pure_inference(x, y, ("T" + "." + str(train_num)))

            # Collect accuracy data
            num_correct += correct
            num_mistakes += mistake
            hamming_loss += list_diff(y_hat, y) / len(y)

            instance_str += ("\t[" + str(num_correct) + "/"
                             + str(train_num + 1) + "]")

            # Print instance details and update training number
            print(instance_str)
            train_num += 1

        # Compute accuracies
        num_examples = train_num
        std_accuracy = num_correct / num_examples
        hamming_accuracy = (1.0 - (hamming_loss / num_examples))

        print()
        print("\t| [Test] Standard accuracy = " + str(std_accuracy * 100) + "%")
        print("\t| [Test] Hamming accuracy = " + str(hamming_accuracy * 100) + "%")
        print()

        # Return testing accuracy
        return hamming_accuracy

    # Inference and (weight) update methods

    def pure_inference(self, x, y, train_instance):
        """
        Do pure inference without any weight update; primarily used in testing
        """

        # NOTE: There be lots of recycled code in these parts a' town

        # Predict (i.e. run inference)
        h = lambda y_partial: self.get_score(x, y_partial)
        bs = self.search_obj(self.b, h, y, self.alphabet)
        y_hat = bs.search()

        # Accuracy stuff
        num_right_chars = len(y) - list_diff(y_hat, y)
        mistake = 0
        correct = 0

        instance_str = ("Data instance " + train_instance)
        result_char = ""

        # If error, update weights
        if y_hat != y:
            result_char = "-"
            mistake = 1

        else:
            result_char = "+"
            correct = 1

        instance_str = ("\t[" + result_char + "]\t" + instance_str + "\t(" + \
                        str(num_right_chars) + "/" + str(len(y)) + ")")

        return y_hat, correct, mistake, num_right_chars, instance_str

    def standard_update(self, x, y, train_instance):
        """
        [ Standard update doesn't converge because it doesn't guarentee violation ]
        Search error:   Maximum scoring terminal output at end of search is wrong
        Weight update:  Standard structured perceptron on highest scoring terminal
        Beam update:    None
        """

        # Predict (i.e. run inference)
        h = lambda y_partial: self.get_score(x, y_partial)
        bs = self.search_obj(self.b, h, y, self.alphabet)
        y_hat = bs.search()

        # Collect accuracy
        num_right_chars = len(y) - list_diff(y_hat, y)
        mistake = 0
        correct = 0

        instance_str = ("Data instance " + train_instance)
        result_char = ""

        # Show real output and predicted output!
        # NOTE: That weird-looking list comprehension below just indicates
        # which characters y_hat got wrong w.r.t. y
        err_display = "\n"
        err_display += ("\t\t\t" + " " + "".join(\
              ["_" if y_hat[i] != y[i] else " " for i in range(len(y))]) \
              + " \n")
        err_display += ("\t\t\t" + "'" + "".join(y_hat).upper() + "'\n")
        err_display += ("\t\t\t" + "'" + "".join(y).upper() + "'" + "*\n")
        err_display += ("\n")

        # If error, update weights
        if y_hat != y:

            # Compute joint-features of correct and predicted outputs
            ideal_phi = self.phi(x, y)
            pred_phi = self.phi(x, y_hat)

            # Perform weight update
            self.w = np.add(self.w, np.dot(self.eta, \
                (np.subtract(ideal_phi, pred_phi))))

            result_char = "-"
            mistake = 1

        else:
            result_char = "+"
            correct = 1

        instance_str = ("\t[" + result_char + "]\t" + instance_str + "\t(" + \
                        str(num_right_chars) + "/" + str(len(y)) + ")")

        return y_hat, correct, mistake, num_right_chars, instance_str, err_display

    def early_update(self, x, y, train_instance):
        """
        [ When correct label falls off the beam (via pruning), update right then ]
        Search error:   No target nodes in beam
        Weight update:  Standard structured perceptron (with recent y_select)
        Beam update:    Reset beam with intial state (or discontinue search)
        """

        # Initialize beam search tree
        h = lambda y: self.get_score(x, y)
        bs = self.search_obj(self.b, h, y, self.alphabet)
        y_detour = None
        y_matching = None

        # Beam search until target node gets pruned
        while bs.expand():

            # Error: target node has been pruned
            if not bs.contains_target():

                # Only want earliest error
                if y_detour is not None: continue

                # NOTE: We use last y-select as node we use for
                # weight update; y-detour denotes the node which led us
                # to expand and remove a target node from the beam
                y_detour = bs.rank(*bs.beam)[-1] #bs.y_select
                y_matching = y[:len(y_detour)]

                # NOTE: Wait until terminal node is reached before performing
                # weight update with the previous two structured outputs; note
                # that there is no way of acheiving a correct output at this
                # point (because no target node in beam!)

        # Predict (i.e. run inference)
        y_hat = bs.complete_max_in_beam()
        num_right_chars = len(y) - list_diff(y_hat, y)
        mistake = 0
        correct = 0

        # ----------------

        instance_str = ("Data instance " + train_instance)
        result_char = ""

        # Show real output and predicted output!
        err_display = self.give_err_bars(y, y_hat)

        if y_hat != y:

            result_char = "-"
            mistake = 1

            # NOTE: We only update when there is a detouring parent, but not
            # when we pick the incorrect terminating node
            if y_detour is not None:

                # Perform weight update
                ideal_phi = self.phi(x, y_matching)
                pred_phi = self.phi(x, y_detour)
                self.w = np.add(self.w, np.dot(self.eta, \
                    (np.subtract(ideal_phi, pred_phi))))

        else:
            result_char = "+"
            correct = 1

        instance_str = ("\t[" + result_char + "]\t" + instance_str + "\t(" + \
                        str(num_right_chars) + "/" + str(len(y)) + ")")

        return y_hat, correct, mistake, num_right_chars, instance_str, err_display

    def max_violation_update(self, x, y, train_instance):
        """
        [ Use maximum violating prefix in search space to do weight update ]
        Search error:   None
        Weight update:  Standard structured perceptron (with max-violating y_partial)
        Beam update:    Reset beam with intial state (or discontinue search)
        """

        # Initialize beam search tree
        h = lambda y: self.get_score(x, y)
        bs = self.search_obj(self.b, h, y, self.alphabet)
        beam_history = []

        # Beam search until we hit any terminal node
        while bs.expand():
            beam_copy = copy.deepcopy(bs.beam)
            beam_history.append(beam_copy)

        # Review each beam b_t in the beam history
        best_prefix_pairs = []
        for t, b_t in enumerate(beam_history):

            # Maximally scoring non-target node, as well as target node which
            # represents (correct) prefix of same length
            max_node_len = len(max(b_t, key = len))
            present_target = y[:max_node_len] # Prefix at time t
            non_target_nodes = [node for node in b_t \
                                if node != present_target \
                                and len(node) == len(present_target)]

            # Ff we only have non-target nodes in the beam, then we've
            # gotta skip this b_t!
            if len(non_target_nodes) == 0: continue

            best_non_target = max(non_target_nodes, key = bs.score)

            # NOTE: Does sign matter here?
            violation = abs(bs.score(present_target) - \
                bs.score(best_non_target))
            best_prefix_pairs.append((present_target, best_non_target, \
                violation))

            # NOTE: Due to the nature of our beam search, we will have only
            # one target node (for any particular tree) of some arbitrary
            # prefix length t

        # Return maximum in best_prefix_pairs w.r.t. violation
        # TODO: Address the more important question of why we have no
        # best_prefix_pairs sometimes
        no_max_violating = False
        if len(best_prefix_pairs) == 0:
            no_max_violating = True
        else: max_violating = max(best_prefix_pairs, key = lambda pair: pair[2])

        # ----------------

        # Predict (i.e. run inference)
        y_hat = bs.complete_max_in_beam()
        num_right_chars = len(y) - list_diff(y_hat, y)
        mistake = 0
        correct = 0

        instance_str = ("Data instance " + train_instance)
        result_char = ""

        # Show real output and predicted output!
        err_display = self.give_err_bars(y, y_hat)

        # Do weight update upon prediction error
        if y_hat != y:
            result_char = "-"
            mistake = 1

            if not no_max_violating:
                # Perform weight update with maximum violating prefix we've
                # ever seen in the beam search
                ideal_phi = self.phi(x, max_violating[0])
                pred_phi = self.phi(x, max_violating[1])
                self.w = np.add(self.w, np.dot(self.eta, \
                    (np.subtract(ideal_phi, pred_phi))))

        else:
            result_char = "+"
            correct = 1

        instance_str = ("\t[" + result_char + "]\t" + instance_str + "\t(" + \
            str(num_right_chars) + "/" + str(len(y)) + ")")

        return y_hat, correct, mistake, num_right_chars, instance_str, err_display

    def rgs(self, x, len_y):
        """
        Randomized Greedy Search (RGS) inference:
        Try and use the current weights to arrive at the correct label;
        we will always return our best guess
        """

        # NOTE: RGS performs terribly and takes way to much time :(

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

    # Phi methods

    def phi_unary(self, x, y):
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

    def phi_pairwise(self, x, y):
        """
        Pairwise joint-feature function:
            0. Do unary features
            1. Capture all two-char permutations
            2. Assign these permuations consistent indices
            3. Count frequencies of each permuation and update vector
               at that index
        """

        # Initial setting of phi dimensions
        self.pairwise_base_index = self.len_x * self.len_y
        dimen = (self.len_x * self.len_y) + (self.len_y ** 2)

        vect = np.zeros((dimen))
        alpha_list = list(self.alphabet)
        alpha_list.sort()

        # (One-time) Generate pair-index object
        if len(self.pairs) == 0:
            for a in alpha_list:
                for b in alpha_list:
                    p = a + b
                    self.pairs.append(p)

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
            comb_index = self.pairs.index(p)
            vect_index = self.pairwise_base_index + comb_index

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

        # Initial setting of phi dimensions
        self.pairwise_base_index = self.len_x * self.len_y
        self.triplet_base_index = self.pairwise_base_index + (self.len_y ** 2)
        dimen = self.triplet_base_index + (self.len_y ** 3)

        vect = np.zeros((dimen))
        alpha_list = list(self.alphabet)
        alpha_list.sort()

        # (One-time) Generate pair and triplet lists
        if len(self.triplets) == 0:
            for a in alpha_list:
                for b in alpha_list:
                    # Grab pair
                    p = a + b
                    self.pairs.append(p)

                    for c in alpha_list:
                        # Grab triplet
                        t = a + b + c
                        self.triplets.append(t)

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
            comb_index = self.pairs.index(p)
            vect_index = self.pairwise_base_index + comb_index

            # Update occurace of pair
            vect[vect_index] += 1

        # Third-order features
        for i in range(len(y) - 2):

            # Get pair index
            a = y[i]
            b = y[i + 1]
            c = y[i + 2]
            t = a + b + c
            comb_index = self.triplets.index(t)
            vect_index = self.triplet_base_index + comb_index

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

        # Initial setting of phi dimensions
        self.pairwise_base_index = self.len_x * self.len_y
        self.triplet_base_index = self.pairwise_base_index + (self.len_y ** 2)
        self.quadruplet_base_index = self.triplet_base_index + (self.len_y ** 3)
        dimen = self.quadruplet_base_index + (self.len_y ** 4)

        vect = np.zeros((dimen))
        alpha_list = list(self.alphabet)
        alpha_list.sort()

        # (One-time) Generate pair, triplet, and quadruplet lists
        if len(self.quadruplets) == 0:
            for a in alpha_list:
                for b in alpha_list:
                    # Grab pair
                    p = a + b
                    self.pairs.append(p)

                    for c in alpha_list:
                        # Grab triplet
                        t = a + b + c
                        self.triplets.append(t)

                        for d in alpha_list:
                            # Grab quadruplet
                            q = a + b + c + d
                            self.quadruplets.append(q)

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
            comb_index = self.pairs.index(p)
            vect_index = self.pairwise_base_index + comb_index

            # Update occurace of pair
            vect[vect_index] += 1

        # Third-order features
        for i in range(len(y) - 2):

            # Get pair index
            a = y[i]
            b = y[i + 1]
            c = y[i + 2]
            t = a + b + c
            comb_index = self.triplets.index(t)
            vect_index = self.triplet_base_index + comb_index

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
            comb_index = self.quadruplets.index(q)
            vect_index = self.quadruplet_base_index + comb_index

            # Update occurace of quadruplet
            vect[vect_index] += 1

        return vect

    # Class utility methods

    def converged(self, diff):
        """
        Given difference between two numbers, determine if they are
        within the convergence range
        """
        diff = int(diff * 100)
        if diff <= self.convergence_range: return True
        return False

    def get_score(self, x, y_hat):
        """
        Compute score of joint-feature function with weights,
        while also setting the weight dimensions to phi dimensions
        dynamically
        """

        # Joint-feature function of predicted label-group for input x
        pred_phi = self.phi(x, y_hat)

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

    def give_err_bars(self, y, y_hat):
        """ Show error bars above incorrect chars in y_hat """

        # NOTE: Assuming all chars in alphabet are same length
        char_len = len(list(self.alphabet)[0])

        err_display = "\n"
        err_display += ("\t\t\t" + " " + "".join(\
              [("_" * char_len) if y_hat[i] != y[i] else (" " * char_len) \
                for i in range(len(y))]) + " \n")
        err_display += ("\t\t\t" + "'" + "".join(y_hat).upper() + "'\n")
        err_display += ("\t\t\t" + "'" + "".join(y).upper() + "'" + "*\n")
        err_display += ("\n")

        return err_display

    def display_header(self, D):
        print()
        print("Structured Perceptron:")
        print()
        print("\tData length (with limitation) = " + str(len(D)))
        print("\tNumber of restarts = " + str(self.R))
        print("\tUpdate method = " + str(self.update_method))
        print("\tSearch object = " + str(self.search_obj))
        print("\tBeam width = " + str(self.b))
        print("\tLearning rate = " + str(self.eta))
        print("\tMax iteration count = " + str(self.MAX))
        print("\tOrder of joint-features = " + str(self.phi_order))
        print("\tAlphabet length = " + str(self.len_y))
        print()

class BeamSearch:
    """ Parent class for basic beam search """

    def __init__(self, b, h, y, alphabet):
        """ Construct beam properties necessary for heuristic search """

        # Beam as a list
        self.beam = []

        # Beam width
        self.b = b

        # Heuristic guiding beam search; tells us desirability of inputted
        # node (as real value) given it's structured data (as a list)
        self.h = h

        # Length of terminal output
        self.term_len = len(y)

        # Store correct output (to know target nodes)
        self.y = y

        # Stores selected y-node which created most recent expansion
        self.y_select = []

        # Alphabet used for constructing nodes in the search space
        self.alphabet = alphabet

    def expand(self):
        """
        Perform single expansion from beam; return false if terminal node is hit,
        otherwise return true

        NOTE: Child inheritor should override this method
        """
        raise NotImplementedError

    def search(self):
        """
        Move through search space guided by given heuristic h, stopping
        search once one node in beam is of given terminal length
        """

        # Loop until complete structure output found in beam
        while self.expand(): pass

        # Get highest scoring, complete output in beam (and return)
        y_hat = self.complete_max_in_beam()

        return y_hat

    def reset(self):
        """ Reset beam back to initial state (i.e. empty beam) """
        self.beam = []

    def complete_max_in_beam(self):
        """ Get max scoring, complete output in beam """

        self.beam = [y for y in self.beam if len(y) == self.term_len]
        y_hat = self.max_in_beam()
        return y_hat

    def _reduce_beam(self, func):
        """ Apply some function to the beam and get some corresponding node """

        # Use dictionary of beam, mapping (encoded) y to it's score
        beam_node_scores = {self.encode_y(y):self.h(y) for y in self.beam}
        beam_subject = func(beam_node_scores, key = \
            (lambda y: beam_node_scores[y]), default = [])
        return self.decode_y(beam_subject)

    def min_in_beam(self):
        """ Return minimum scoring node in beam """
        return self._reduce_beam(min)

    def max_in_beam(self):
        """ Return maximum scoring node in beam w.r.t. heuristic h """
        return self._reduce_beam(max)

    def contains_target(self):
        """ Boolean returned indicating if beam contains a target node """

        # Check if any node in beam is a subset of correct y
        y_str = self.encode_y(self.y)
        for y_node in self.beam:
            y_node_str = self.encode_y(y_node)
            if y_str.find(y_node_str) == 0: return True
        return False

    def gen_children(self, y):
        """
        Using alphabet and given (partial) labelling y, generate list of
        children, each with 1-char difference from parent
        """

        # Don't expand if we can't
        if len(y) == self.term_len: return []

        children = []
        for char in self.alphabet:
            child = y + [char]
            children.append(child)

        return children

    def rank(self, *inputs):
        """
        Rank the given inputs according to our current heuristic from
        lowest to highest
        """
        sorted_inputs = sorted(inputs, key = lambda y: self.h(y))[:self.b]
        return sorted_inputs

    def score(self, y):
        """ Use heuristic to return score of given input y """
        return self.h(y)

    @staticmethod
    def encode_y(y):
        """ Encode y from list to string """
        return "-".join(y)

    @staticmethod
    def decode_y(y):
        """ Decode y from string to list """
        if y == []: return []
        return y.split("-")

    @staticmethod
    def give_target_nodes(arr, y):
        """ Give proper prefixes of complete output y in given array """

        encoded_arr = list(map(lambda item: \
            BeamSearch.encode_y(item), arr))
        encoded_y = BeamSearch.encode_y(y)
        target_nodes = []
        for item in encoded_arr:
            if encoded_y.find(item) == 0:
                decoded_item = BeamSearch.decode_y(item)
                target_nodes.append(decoded_item)
        return target_nodes

class BestFirstBeamSearch(BeamSearch):
    def __init__(self, b, h, y, alphabet):
        super().__init__(b, h, y, alphabet)

    @overrides(BeamSearch)
    def expand(self):
        """
        Expand best-scoring node in beam; return false if terminal node is hit,
        otherwise return true
        """

        # [Grab] maximum scoring output in beam
        self.y_select = self.max_in_beam()

        # [Expand] upon maximum scoring (partial) output
        # NOTE: We don't expand beyond terminal length
        candidates = self.beam + self.gen_children(self.y_select)
        if len(self.y_select) > 0: candidates.remove(self.y_select)

        # [Prune] excess, lower-scoring nodes; reverse sorting, hence
        # negating the value from our heuristic function; randomly
        # shuffle candidates (before sorting) to give equally scoring
        # nodes a fair chance
        random.shuffle(candidates)
        self.beam = sorted(candidates, key = lambda y: -self.h(y))[:self.b]

        # [Check] for terminal node/output
        for out in self.beam:
            if len(out) == self.term_len: return False
        return True

class BreadthFirstBeamSearch(BeamSearch):
    def __init__(self, b, h, y, alphabet):
        super().__init__(b, h, y, alphabet)

    @overrides(BeamSearch)
    def expand(self):
        """
        Expand every node in beam; return false if terminal node is hit,
        otherwise return true
        """

        # [Expand] upon every (partial) output in beam
        # NOTE: Implicitly only keeping children of expanded parents, leaving
        # parents behind in the dust
        candidates = []
        for y_partial in self.beam:
            candidates += self.gen_children(y_partial)
        if len(self.beam) == 0: candidates = self.gen_children(self.beam) # init

        # [Prune] excess, lower-scoring nodes - maintain beam width
        random.shuffle(candidates)
        self.beam = sorted(candidates, key = lambda y: -self.h(y))[:self.b]

        # [Check] for terminal node/output
        for out in self.beam:
            if len(out) == self.term_len: return False
        return True
