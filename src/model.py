# model.py

"""
STRUCTURED PERCEPTRON

Methods immediately relevant to the concept of a perceptron
"""

def ospt(D, phi, R, eta, MAX, L, w = None):
    """
    Online structured perceptron training/testing:
        1. If weight vector w is not supplied, we're training;
        2. Else, we're testing
    """

    # See if we're training or testing
    training = w is None
    if training: work_word = "Train"
    else: work_word = "Test"

    # Check data limit
    if L is None: L = len(D)

    # Display heading
    print()
    print("<" + work_word + "> ", end = "")
    print("Structured Perceptron:")
    print()
    print("\tData length (with limitation) = " + str(L))
    print("\tNumber of restarts = " + str(R))
    print("\tLearning rate = " + str(eta))
    print("\tMax iteration count = " + str(MAX))
    print("\tNumber of joint-features = " + str(phi_dimen))
    print("\tAlphabet length = " + str(len(alphabet)))
    print()

    # Record model's progress w.r.t. accuracy (and iteration improvment)
    it_improvement = np.zeros((L))
    acc_progress = []
    if training: pw = pg.plot()

    # Setup weights of scoring function to 0, if weight vector is
    # not supplied
    if training: w = np.zeros((phi_dimen))

    # Iterate until max iterations or convergence
    for it in range(MAX):

        # Time each iteration
        it_start = time.clock()

        print("[Iteration " + str(it) + "]\n")

        # Essential iteration-related vars
        train_num = 0
        num_mistakes = 0
        num_correct = 0

        # Go through training examples
        for x, y in D[:L]:

            # Skip empty data points
            if len(x) < 1: continue

            # Predict
            y_hat = rgs(x, phi, w, R, len(y))
            num_right_chars = len(y) - list_diff(y_hat, y)

            # If error, update weights
            instance_str = (work_word + " instance " + str(it)
                            + "." + str(train_num))
            if y_hat != y:
                instance_str = ("\t[-]\t" + instance_str + "\t("
                                + str(num_right_chars) + "/" + str(len(y)) + ")")
                if training:
                    # TODO: Uncomment the weight update!
                    w = np.add(w, np.dot(eta, (np.subtract(phi(x, y),
                                                           phi(x, y_hat)))))
                    # HACK: Setting "weights" each time might be too much
                    weights = w
                    num_mistakes += 1
            else:
                instance_str = ("\t[+]\t" + instance_str + "\t(" + str(len(y))
                                + "/" + str(len(y)) + ")")
                num_correct += 1

            instance_str += ("\t[" + str(num_correct) + "/"
                             + str(train_num + 1) + "]")

            # Measure iteration improvment (compared to last)
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
    if training: return save_w(w)
    return

def rgs(x, phi, w, R, len_y):
    """
    Randomized Greedy Search (RGS) inference:
    Try and use the current weights to arrive at the correct label;
    we will always return our best guess
    """

    for i in range(R):

        # Initialize best scoring output randomly
        y_hat = get_random_y(len_y)

        # Until convergence
        while True:

            # Get max char
            y_max = get_max_one_char(w, phi, x, y_hat)
            if y_max == y_hat: break
            y_hat = y_max

    return y_hat

def phi_unary(x, y, len_x = None, len_y = None):
    """ Unary joint-feature function """

    if len_x is None: dimen = phi_dimen
    else: dimen = len_x * len_y
    vect = np.zeros((dimen))

    for i in range(len(x)):

        x_i = x[i]
        y_i = y[i]

        # Sorting keeps consistency of indices with respect to
        # all prior and following phi(x, y) vectors
        alpha_list = list(alphabet)
        alpha_list.sort()
        index = alpha_list.index(y_i)
        x_vect = np.array(x_i)

        # Manual insertion of x into standard vector
        # NOTE: Holy fuck, had "= x_vect[j]" before, not "+="
        y_target = len(x_i) * index
        for j in range(len(x_i)): vect[j + y_target] += x_vect[j]

    return vect

def phi_pairwise(x, y, len_x = None, len_y = None):
    """
    Pairwise joint-feature function:
        0. Do unary features
        1. Capture all two-char permutations
        2. Assign these permuations consistent indices
        3. Count frequencies of each permuation and update vector
           at that index
    """

    global pairwise_base_index, pairs

    # NOTE: len_y = len(alphabet)
    # Initial setting of phi dimensions
    if len_x is None: dimen = phi_dimen
    else:
        pairwise_base_index = len_x * len_y
        dimen = (len_x * len_y) + (len_y ** 2)

    vect = np.zeros((dimen))
    alpha_list = list(alphabet)
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

def phi_third_order(x, y, len_x = None, len_y = None):
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

    # NOTE: len_y = len(alphabet)
    # Initial setting of phi dimensions
    if len_x is None: dimen = phi_dimen
    else:
        pairwise_base_index = len_x * len_y
        triplet_base_index = pairwise_base_index + (len_y ** 2)
        dimen = triplet_base_index + (len_y ** 3)

    vect = np.zeros((dimen))
    alpha_list = list(alphabet)
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

        # Update occurace of pair
        #print(t, "occurs at", vect_index)
        vect[vect_index] += 1

    return vect

def phi_fourth_order(x, y, len_x = None, len_y = None):
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

    # NOTE: len_y = len(alphabet)
    # Initial setting of phi dimensions
    if len_x is None: dimen = phi_dimen
    else:
        pairwise_base_index = len_x * len_y
        triplet_base_index = pairwise_base_index + (len_y ** 2)
        quadruplet_base_index = triplet_base_index + (len_y ** 3)
        dimen = quadruplet_base_index + (len_y ** 4)

    vect = np.zeros((dimen))
    alpha_list = list(alphabet)
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

        # Update occurace of pair
        #print(t, "occurs at", vect_index)
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

        # Update occurace of pair
        #print(q, "occurs at", vect_index)
        vect[vect_index] += 1

    return vect

def get_score(w, phi, x, y_hat):
    return np.dot(w, phi(x, y_hat))

def get_random_y(len_y):
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
        rand_char = random.sample(alphabet, 1)[0]
        rand_y.append(rand_char)

    return rand_y

def get_max_one_char(w, phi, x, y_hat):
    """
    Make one-character changes to y_hat, finding which
    single change produces the best score; we return
    that resultant y_max
    """

    # Initialize variables to max
    s_max = get_score(w, phi, x, y_hat)
    y_max = y_hat

    for i in range(len(y_hat)):

        # Copy of y_hat to play with
        y_temp = copy.deepcopy(y_hat)

        # Go through a-z at i-th index
        for c in alphabet:

            # Get score of 1-char change
            y_temp[i] = c
            s_new = get_score(w, phi, x, y_temp)

            # Capture highest-scoring change
            if s_new > s_max:
                s_max = s_new
                y_max = y_temp

    return y_max
