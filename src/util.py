# util.py

import os

"""
UTILITY FUNCTIONS
Methods not directly relevant to the concept of the structured
percpetron, but more to the maintainance and assistance of
more basic computation in program
"""

def write_report(report):
    """
    Append single report to report file as CSV, allowing for data to
    be analyzed as a spreadsheet
    """

    # NOTE: We already know the exact contents of a report from main()
    # Convert all data in report to a string!
    report = list(map(str, report))
    report_file = "out/reports/report.csv"
    with open(report_file, "a") as f:

        # Compile report as CSV
        report_str = report[0] + "," + report[4] + "," + report[5] \
            + "," + report[6] + "," + report[1] + "," + report[2] \
            + "," + report[7] + "\n"

        # Append to file
        f.write(report_str)

# Custom annotation to denote child overriding method of parent
def overrides(interface_class):
    def overrider(method):
        assert(method.__name__ in dir(interface_class))
        return method
    return overrider

def dprint(s):
    if verbose: print(s)

def setify(num):
    """
    Set-ify that number (i.e. remove trailing zeros, as is
    automatically done in the alphabet set) for consistency
    """

    return list(set([num]))[0]

def beep():
    """ Give absent-minded programmer a notification """

    freq = 700  # Hz
    dur = 1000  # ms
    for i in range(10):
        winsound.Beep(freq, dur)
        freq += 100
    winsound.Beep(freq * 2, dur * 4) # RRRREEEEEEEEEEE!

def list_diff(a, b):
    """ Show's degree of difference of list a from b """

    if len(a) != len(b):
        raise ValueError("Lists of different length.")
    return sum(i != j for i, j in zip(a, b))

def give_sign(n):
    if n < 0: return str(n) # Number will already be negative
    if n > 0: return "+" + str(n)
    return str(n) # We'll say 0 has no sign

def save_w(w):
    """ Serialize the weights of perceptron into local file """

    # TODO: Encode filenames with the following:
    #   0. Number of iterations (until stopped or converged)
    #   1. Degree of phi
    #   2. Type of data (e.g. nettalk or ocr)
    #   3. Version (i.e. the number of weight files)

    print("-" * 35)
    print("Saving weights to local file...")

    # Get user's attention
    if windows: beep()

    # Ask if they want to save
    if (input("Proceed? (y/n): ").strip() == "n"): exit(0)

    # List weight files
    files = [f for f in os.listdir(weights_dir)
         if os.path.isfile(f)
         and f.split(".")[-1] == "npy"]
    print("\tCurrent weight files (to avoid conflicts):", files)

    # Enter filename and save
    w_file_name = input("\tPlease enter filename: ")
    np.save(weights_dir + w_file_name, w)
    print("Saved!")
    print("-" * 35)

    return w

def load_w():
    """ Deserialize weights of perceptron from local file """

    print("-" * 35)
    print("Loading weights from local file:")

    # Show available weight files
    files = [f for f in os.listdir(weights_dir)
             if os.path.isfile(f)
             and f.split(".")[-1] == "npy"]
    print("\tAvailable weight files:", files)

    # User selects file (looping until valid file) -> we load
    w_file_name = ""
    while w_file_name not in files:
        w_file_name = input("\tPlease enter filename: ")
    w = np.load(weights_dir + w_file_name)
    print("File loaded!")
    print("-" * 35)

    return w

def signal_handler(signal, frame):
    """ Save weights on Ctrl+C """

    global sig

    # Check for the double Ctrl+C, which means exit
    if sig is True: exit(0)

    sig = True
    print('[Ctrl+C pressed]')
    save_w(weights)
    sig = False
    exit(0)

def set_list(l, i, v):
    try:
        l[i] = v
    except IndexError:
        for _ in range(i - len(l) + 1):
            l.append(None)
        l[i] = v

def get_data_files(data_dir):
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

def parse_data_file(file_loc):
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
