# data.py

class Data:
    """
    Collects and parses training and testing sets from directory hierarchy 
    """

    class Set:
        def __init__(self, name, train, test, alphabet, len_x):
            self.__name__ = name
            self.train = train      # Training set
            self.test = test        # Testing set
            self.alphabet = alphabet
            self.len_x = len_x

    def __init__(self, data_dir):

        # Get our raw training and testing data
        self._data_dir = data_dir
        print("Grabbing data from {}...".format(self._data_dir), flush = True)
        self._data_files = Data.get_files(self._data_dir)

        # Parse train & test data
        print("Parsing training/testing data...", flush = True)
        self.parsed = []
        for raw_train, raw_test in self._data_files:
            train, len_x, alphabet = Data.parse_file(raw_train)
            test, *_ = Data.parse_file(raw_test)
            parsed_set = self.Set(raw_test, train, test, alphabet, len_x)
            self.parsed.append(parsed_set)

    @staticmethod
    def get_files(data_dir):
        """
        Recursively get all data files in given directory,
        returning tuples of (train, test) pairs
        """

        data_files = []

        for root, subdirs, files in os.walk(data_dir):

            # Recurse on subdirectories
            for subdir in subdirs:
                data_files + get_data_files(subdir)

            # Collect (train.txt. test.txt)
            train_test_pair = []
            for filename in files:
                file_path = os.path.join(root, filename)

                if filename.split(".")[0] == "train":
                    set_list(train_test_pair, 0, file_path)
                if filename.split(".")[0] == "test":
                    set_list(train_test_pair, 1, file_path)

            if len(train_test_pair) != 0:
                data_files.append(tuple(train_test_pair))

        return data_files

    @staticmethod
    def parse_file(file_loc):
        """ Parse raw data into form of [(x_0, y_0), ..., (x_n, y_n)] """

        # Set of given dataset's alphabet of labels
        alphabet = set()

        data_arr = []
        len_x_vect = -1

        with open(file_loc) as f:

            x = []
            y = []

            # Take collection of examples (e.g. collection of pairs of
            # character data x_i matched with the actual character
            # class y_i) and push into data array
            for line in f:

                # Push single collection of examples onto data array
                # when newline is encountered
                if not line.strip():
                    data_arr.append((x, y))
                    x = []
                    y = []
                    continue

                # Parse one example (x_i, y_i)
                l_toks = line.split("\t")
                x_i_str = l_toks[1][2:] # Trim leading "im" tag
                x_i = [int(c) for c in x_i_str]
                y_i = setify(l_toks[2])

                # Collect length of all x_i's
                if len_x_vect < 0: len_x_vect = len(x_i)

                # Take note of all possible labels (i.e. the set Y)
                # NOTE: Listifying y_i is necessary to keep leading
                # zeroes, e.g. maintaining '04' rather than '4'
                alphabet.update([y_i])

                # Add single example to collection
                x.append(x_i)
                y.append(y_i)

        return data_arr, len_x_vect, alphabet
