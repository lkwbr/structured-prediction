# search.py

"""
BEAM SEARCH
Home to implementations of Breadth-First and Best-First Beam Search
"""

from util import *

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
