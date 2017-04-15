# util.py

"""
UTILITY FUNCTIONS
Methods not directly relevant to the concept of the structured
percpetron, but more to the maintainance and assistance of
more basic computation in program
"""

import os
import sys
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

_stdout = None

def send_email(email, epass, esubject, ebody):
    """ Send an email """

    fromaddr = "{}".format(email)
    toaddr = "{}".format(email)
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = esubject

    body = ebody
    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, "{}".format(epass))
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()

def disable_stdout():
    global _stdout
    _stdout = sys.stdout
    sys.stdout = open('out/tmp', 'w')

def enable_stdout():
    sys.stdout = _stdout

def write_report(report, filename):
    """
    Append single report to report file as CSV, allowing for data to
    be analyzed as a spreadsheet
    """

    # NOTE: We already know the exact contents of a report from main()
    # Convert all data in report to a string!
    report = list(map(str, report))
    report_file = "out/reports/" + filename + ".csv"
    with open(report_file, "a") as f:

        # Compile report as CSV
        report_str = report[0] + "," + report[4] + "," + report[5] \
            + "," + report[6] + "," + report[1] + "," + report[2] \
            + "," + report[7] + "\n"

        # Append to file
        f.write(report_str)

def give_err_bars(alphabet, y, y_hat):
    """ Show error bars above incorrect chars in y_hat """

    # NOTE: Assuming all chars in alphabet are same length
    char_len = len(list(alphabet)[0])

    err_display = "\n"
    err_display += ("\t\t\t" + " " + "".join(\
          [("_" * char_len) if y_hat[i] != y[i] else (" " * char_len) \
            for i in range(len(y))]) + " \n")
    err_display += ("\t\t\t" + "'" + "".join(y_hat).upper() + "'\n")
    err_display += ("\t\t\t" + "'" + "".join(y).upper() + "'" + "*\n")

    return err_display

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
