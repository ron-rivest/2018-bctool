import csv
import numpy as np

##############################################################################
## Random number generation
##############################################################################

# This function is taken from audit-lab directly.
def convert_int_to_32_bit_numpy_array(v):
    """
    Convert value v, which should be an arbitrarily large python integer
    (or convertible to one) to a numpy array of 32-bit values,
    since this format is needed to initialize a numpy.random.RandomState
    object.  More precisely, the result is a numpy array of type int64,
    but each value is between 0 and 2**32-1, inclusive.

    Example: input 2**64 + 5 yields np.array([5, 0, 1], dtype=int)

    Input Parameters:

    -v is an integer, representing the audit seed that's being
    passed in. We expect v to be non-negative.

    Returns:

    -numpy array created deterministically from v that will
    be used to initialize the Numpy RandomState.
    """

    v = int(v)
    if v < 0:
        raise ValueError(("convert_int_to_32_bit_numpy_array: "
                          "{} is not a nonnegative integer, "
                          "or convertible to one.").format(v))
    v_parts = []
    radix = 2 ** 32
    while v > 0:
        v_parts.append(v % radix)
        v = v // radix
    # note: v_parts will be empty list if v==0, that is OK
    return np.array(v_parts, dtype=int)


def create_rs(seed):
    """
    Create and return a Numpy RandomState object for a given seed.
    The input seed should be a python integer, arbitrarily large.
    The purpose of this routine is to make all the audit actions reproducible.

    Input Parameters:

    -seed is an integer or None. Assuming that it isn't None, we
    convert it into a Numpy Array.

    Returns:

    -a Numpy RandomState object, based on the seed, or the clock
    time if the seed is None.
    """

    if seed is not None:
        seed = convert_int_to_32_bit_numpy_array(seed)
    return np.random.RandomState(seed)

def preprocess_csv(path_to_csv):
    """
    Preprocess a CSV file into the correct format for our
    sample tallies. In particular, we ignore the county name column
    and summarize the relevant information about the sample tallies
    in each county, the total number of votes in each county, and
    the candidate names.

    Input Parameters:

    -path_to_csv is a string, representing the full path to the CSV
    file, containing sample tallies.

    Returns:

    -sample_tallies is a list of lists. Each list represents the sample tally
    for a given county. So, sample_tallies[i] represents the tally for county
    i. Then, sample_tallies[i][j] represents the number of votes candidate
    j receives in county i.

    -total_num_votes is an integer representing the number of
    ballots that were cast in this election.

    -candidate_names is an ordered list of strings, containing the name of
    every candidate in the contest we are auditing.
    """

    with open(path_to_csv) as csvfile:
        sample_tallies = []
        total_num_votes = []
        reader = csv.DictReader(csvfile)
        candidate_names = [col for col in reader.fieldnames
                           if col.strip().lower() not in
                           ["county name", "total votes"]]
        for row in reader:
            sample_tally = []
            for key in row:
                if key.strip().lower() == "county name":
                    continue
                if key.strip().lower() == "total votes":
                    total_num_votes.append(int(row[key]))
                else:
                    count = int(row[key].strip())
                    assert count >= 0
                    sample_tally.append(count)
            sample_tallies.append(sample_tally)

    for i, sample_tally in enumerate(sample_tallies):
        assert 0 <= sum(sample_tally) <= total_num_votes[i]

    return sample_tallies, total_num_votes, candidate_names

