# model.py

import pyqtgraph as pg
import numpy as np
import random
import time
import copy

# Personal
from util import *
from search import *
from model import Model

"""
STRUCTURED PERCEPTRON
Methods immediately relevant to the concept of a generalized perceptron
"""

class StructuredPerceptron(Model):

    # NOTE: Any variable declared outside of the constructor is a static
    # variable, and will change all instances of this class if they are changed;
    # however, object variables (declared in __init__) are unique to the object.

    __name__ = "StructuredPerceptron"

    def __init__(self, alphabet, len_x, phi_order, update_method, search_type, \
        R, eta, MAX, b):

        super().__init__(alphabet, len_x)

        # Candy shop - so many choices!
        self.__phi_funcs = [ \
            self.__phi_unary, \
            self.__phi_pairwise, \
            self.__phi_third_order, \
            self.__phi_fourth_order]
        self.__update_methods = [ \
            self.__standard_update, \
            self.__early_update, \
            self.__max_violation_update]
        self.__search_types = [ \
            BestFirstBeamSearch,
            BreadthFirstBeamSearch]

        # Perceptron-related
        self.R = R                          # Number of restarts
        self.eta = eta                      # Learning rate
        self.MAX = MAX                      # Maximum number of iterations
        self.w = None                       # Learned weight vector
        self.update_method = self.__update_methods[update_method]

        # Phi: joint-feature function
        self.phi_order = phi_order
        self.phi = self.__phi_funcs[self.phi_order - 1]
        self.phi_dimen = -1                 # Dimensionality of phi
        self.__pairwise_base_index = -1
        self.__triplet_base_index = -1
        self.__quadruplet_base_index = -1
        self.__pairs = []
        self.__triplets = []
        self.__quadruplets = []

        # Beam search related
        self.b = b
        self.search_obj = self.__search_types[search_type]

    # Main methods: training and testing with provided data

    @overrides(Model)
    def train(self, D):
        """ Train on input data set D """

        # Display heading
        self.__display_header(D)

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
                if self.__converged(ham_acc_it_diff):
                    print("Model has converged:")
                    print("\tAccuracy = " + str(accuracy * 100) + "%")
                    print("\tIteration = " + str(it))
                    break

        # Return training accuracy
        return hamming_accuracy

    @overrides(Model)
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
                self.__pure_inference(x, y, ("T" + "." + str(train_num)))

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

    def __pure_inference(self, x, y, train_instance):
        """
        Do pure inference without any weight update; primarily used in testing
        """

        # NOTE: There be lots of recycled code in these parts a' town

        # Predict (i.e. run inference)
        h = lambda y_partial: self.__get_score(x, y_partial)
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

    def __standard_update(self, x, y, train_instance):
        """
        [ Standard update doesn't converge because it doesn't guarentee violation ]
        Search error:   Maximum scoring terminal output at end of search is wrong
        Weight update:  Standard structured perceptron on highest scoring terminal
        Beam update:    None
        """

        # Predict (i.e. run inference)
        h = lambda y_partial: self.__get_score(x, y_partial)
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

    def __early_update(self, x, y, train_instance):
        """
        [ When correct label falls off the beam (via pruning), update right then ]
        Search error:   No target nodes in beam
        Weight update:  Standard structured perceptron (with recent y_select)
        Beam update:    Reset beam with intial state (or discontinue search)
        """

        # Initialize beam search tree
        h = lambda y: self.__get_score(x, y)
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
        err_display = self.__give_err_bars(y, y_hat)

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

    def __max_violation_update(self, x, y, train_instance):
        """
        [ Use maximum violating prefix in search space to do weight update ]
        Search error:   None
        Weight update:  Standard structured perceptron (with max-violating y_partial)
        Beam update:    Reset beam with intial state (or discontinue search)
        """

        # Initialize beam search tree
        h = lambda y: self.__get_score(x, y)
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
        err_display = self.__give_err_bars(y, y_hat)

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

    def __rgs(self, x, len_y):
        """
        Randomized Greedy Search (__rgs) inference:
        Try and use the current weights to arrive at the correct label;
        we will always return our best guess
        """

        # NOTE: __rgs performs terribly and takes way to much time :(

        for i in range(self.R):

            # Initialize best scoring output randomly
            y_hat = self.__get_random_y(len_y)

            # Until convergence
            while True:

                # Get max char
                y_max = self.__get_max_one_char(x, y_hat)
                if y_max == y_hat: break
                y_hat = y_max

        return y_hat

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

    # Class utility methods

    def __get_score(self, x, y_hat):
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

    def __get_random_y(self, len_y):
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

    def __get_max_one_char(self, x, y_hat):
        """
        Make one-character changes to y_hat, finding which
        single change produces the best score; we return
        that resultant y_max
        """

        # Initialize variables to max
        s_max = self.__get_score(x, y_hat)
        y_max = y_hat

        for i in range(len(y_hat)):

            # Copy of y_hat to play with
            y_temp = copy.deepcopy(y_hat)

            # Go through a-z at i-th index
            for c in self.alphabet:

                # Get score of 1-char change
                y_temp[i] = c
                s_new = self.__get_score(x, y_temp)

                # Capture highest-scoring change
                if s_new > s_max:
                    s_max = s_new
                    y_max = y_temp

        return y_max

    def __set_weights(self, w):
        """ Allow the outside to set our scoring function weights """
        self.w = w

    def __give_err_bars(self, y, y_hat):
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

    def __display_header(self, D):
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
