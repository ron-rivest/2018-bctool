# bctool.py
# Authors: Ronald L. Rivest, Mayuri Sridhar, Zara A Perumal
# July 9, 2018
# python3


"""
This module provides support for auditing of a single plurality contest
over multiple jurisdictions using a Bayesian ballot-level
comparison audit.

(See github.com/ron-rivest/2018-bptool/ for similar code for
a Bayesian ballot-level polling audit.  The code here was based
in part on that code.)

This module provides routines for computing the winning probabilities
for various choices, given audit sample data.  Thus, this program
provides a risk-measuring functionality.

More precisely, the code builds a Bayesian model of the unsampled
ballots, given the sampled ballots.  This model is probabilistic,
since there is uncertainty about what the unsampled ballots are.
However, the model is generative: we generate many possible
sets of likely unsampled ballots, and report the probability that
various choices win the contest.  (See References below for
more details.)

The main output of the program is a report on the probability
that each choice wins, in these simulations.

The contest may be single-jurisdiction or multi-jurisdiction.
More precisely, we assume that there are a number of "collections"
of ballots that may be sampled.  Each relevant jurisdiction may
have one or more such collections.  For example, a jurisdiction
may be a county, with one collection for ballots submitted by
mail, and one collection for ballots cast in-person.

This module may be used for ballot-polling audits (where there
no reported choices for ballots) or "hybrid" audits (where some
collections have reported choices for ballots, and some have not).


To use this program from the command line, give a command like

        python3 bctool.py collections.csv reported.csv sample.csv

    where

        python3            is the python3 interpreter on your system
                           (this might be just called "python")
                           Besides being python3, your installation must
                           include the numpy package.  Anaconda provides
                           an excellent installation framework that
                           includes numpy.

        collections.csv    is the path to a file giving the number of (cast)
                           votes in each collection.

        reported.csv       is the path to a file giving the reported number of
                           votes for each choice in each collection.

        sample.csv         is the path to a file giving the number of

                                    (reported choice, actual choice)

                           pairs in the sample for each collection.  Here the
                           reported choice is the choice that the scanner
                           reported to have been on the ballot; the actual
                           choice is the choice revealed by a manual
                           examination of the ballot.


FILE FORMATS are as follows; all are CSV files with one header line, then
a number of data lines.


COLLECTIONS.CSV format:

    Example collections.csv file:

        Collection,      Votes,      Comment
        Bronx,           11000
        Queens,         120000
        Mail-In,         56000

    with one header line, as shown, then one data line for
    each collection of paper ballots that may be sampled.
    Each data line gives the name of the collection and
    then the number of cast votes in that collection
    (that is, the number of cast paper ballots in the collection).
    An additional column for optional comments is provided.


REPORTED.CSV format

    Example reported.csv file:

        Collection,  Reported,    Votes,       Comment
        Bronx,       Yes,          8500
        Bronx,       No,           2500
        Queens,      Yes,        100000
        Queens,      No,          20000
        Mail-In,     Yes,          5990
        Mail-In,     No,          50000
        Mail-In,     Overvote,       10

    with one header line, as shown, then one data line for each
    collection and each reported choice in that collection,
    giving the number of times such reported choice is reported
    to have appeared.  An additional column for optional comments is
    provided.

    A reported choice need not be listed, if it occurred zero times,
    although every possible choice (except write-ins) should be listed
    at least once for some contest, so that this program knows what the
    possible votes are for this contest.

    For each collection, the sum, over reported choices,  of the Votes given
    should equal the Votes value given in the corresponding line in
    the collections.csv file.

    Write-ins may be specified as desired, perhaps with a string
    such as "Write-in:Alice Jones".

    For ballot-polling audits, use a reported choice of "-MISSING"
    or "-noCVR" or any identifier starting with a "-".  (Tagging
    the identifier with an initial "-" prevents it from becoming
    elegible for winning the contest.)


SAMPLE.CSV format:

    Example:

        Collection,  Reported,  Actual,    Votes,       Comment
        Bronx,       Yes,       Yes,         62
        Bronx,       No,        No,          21
        Bronx,       No,        Overvote,     1
        Queens,      Yes,       Yes,        504
        Queens,      No,        No,          99
        Mail-In,     Yes,       Yes,         17
        Mail-In,     No,        No,          73
        Mail-In,     No,        Lizard People, 2
        Mail-In,     No,        Lizard People, 1

    with one header line, as shown then at least one data line for
    each collection for each reported/actual choice pair that was seen
    in the audit sample at least once, giving the number of times it
    was seen in the sample.  An additional column for comments is
    provided.

    There is no need to give a data line for a reported/actual pair
    that wasn't seen in the sample.

    If you give more than one data line for a given collection,
    reported choice, and actual choice combination, then the votes
    are summed. (So the mail-in collection has three ballots the scanner said
    showed "No", but that the auditors said actually showed "Lizard
    People".)  For example, you may give one line per audited ballot,
    if you wish.

    The lines of this file do not need to be in any particular order.
    You can just add more lines to the end of this file as the audit
    progresses, if you like.

    The Reported choices must have appeared in
    the reported.csv file; the actual choices may be anything.

    As the audit progresses, the file sample.csv will change, but the
    other two files should not.  Only the file sample.csv has data
    obtained from manually auditing a random sample of the paper
    ballots.

    It is assumed here that sampling is done without replacement.

    This module does not assume that each collection is sampled at the
    same rate --- the audit might look at 1% of the ballots in the
    Bronx collection while looking at 2% of the ballots in the Queens
    collection.  A Bayesian audit can easily work with such
    differences in sampling rates.

    (The actual sampling rates are presumed to be determined by other
    modules; the present module only computes winning probabilities
    based on the sample data obtained so far.)


There are optional parameters as well, to see their documentation,
give command

    python bctool.py --h

Like github.com/ron-rivest/audit-lab, the Bayesian model defaults
to using a prior pseudocount of

    +1   for each (reported, actual) pair of choices (R, A) where R != A

    +50  for the (reported, actual) pair (R, A) of choices where R == A

These default values may be changed via command-line options.

This module can be used for ballot-polling audits too, if all
reported votes are set to "-MISSING" (or some other value that
starts with an initial "-"; even just "-" will do).  For a
hybrid audit where some collections have CVRs and some do not,
it suffices to use the reported choice "-MISSING" for ballots
in those collections not having CVRs, and to the available
reported choice otherwise.


If this module is imported by another python module,
rather than used stand-alone from the command line, then
the procedure

    compute_win_probs

can compute the desired probability of each choice winning a full
manual tally, given the sample tallies for each collection.


More description of Bayesian auditing methods can be found in:

    A Bayesian Method for Auditing Elections
    by Ronald L. Rivest and Emily Shen
    EVN/WOTE'12 Proceedings (2012)
    http://people.csail.mit.edu/rivest/pubs.html#RS12z

    Bayesian Tabulation Audits: Explained and Extended
    by Ronald L. Rivest
    (2018)
    http://people.csail.mit.edu/rivest/pubs.html#Riv18a

    Bayesian Election Audits in One Page
    by Ronald L. Rivest
    (2018)
    http://people.csail.mit.edu/rivest/pubs.html#Riv18b

"""

import argparse

import csv
import sys

import numpy as np


#
# Utilities
#


def duplicates(L):
    """
    Return a list of the duplicates occurring in a given list L.

    If there are no duplicates, return empty list.
    """

    dupes = list()
    seen = set()
    for x in L:
        if x in seen:
            dupes.append(x)
        seen.add(x)
    return dupes

assert duplicates([1, 2, 1, 3, 1, 4, 2, 5]) == [1, 1, 2]


def convert_to_int_if_possible(x):
    """
    Convert x to int if possible and return it; else just return x.
    """

    try:
        x = int(x)
        return x
    except:
        return x

assert convert_to_int_if_possible("1") == 1
assert convert_to_int_if_possible("A") == "A"


#
# Random number generation
#

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

        v         -- an integer, representing the audit seed that's being
                     passed in. We expect v to be non-negative.

    Returns:

        numpy array  -- created deterministically from v that will
                        be used to initialize the Numpy RandomState.
    """

    v = int(v)
    if v < 0:
        raise ValueError(("convert_int_to_32_bit_numpy_array: "
                          "{} is not a nonnegative integer, "
                          "or convertible to one.")
                         .format(v))
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

        seed              -- an integer or None.
                             Assuming that it isn't None, we
                             convert it into a Numpy Array.

    Returns:

        a Numpy RandomState object  --  based on the seed, or the clock
                                        time if the seed is None.
    """

    if seed is not None:
        seed = convert_int_to_32_bit_numpy_array(seed)
    return np.random.RandomState(seed)


#
# Routines for command-line interface and file (csv) input
#

def parse_args():
    """
    Parse command-line arguments and return resulting 'args' object.

    """

    parser = argparse.ArgumentParser(
        description='Bayesian Comparison Audit Support Program For'
                    'A Single Contest '
                    'Across Multiple Jurisdictions or Collections')

    # REQUIRED POSITIONAL COMMAND-LINE ARGUMENTS

    parser.add_argument(
        "collections_file",
        help="path to CSV file giving collection names and sizes.",
        default="collections.csv")

    parser.add_argument(
        "reported_file",
        help="path to CSV file giving number of times each choice "
             "was reportedly made in each collection.",
        default="reported.csv")

    parser.add_argument(
        "sample_file",
        help="path to CSV file giving number of times each "
             "(reported choice, actual choice) pair was seen in "
             "each collection in the audit sample.",
        default="sample.csv")

    # OPTIONAL COMMAND-LINE ARGUMENTS

    parser.add_argument("--audit_seed",
                        help="For reproducibility, we provide the option to "
                             "seed the randomness in the audit. If the same "
                             "seed is provided, the audit will return the "
                             "same results.",
                        type=int,
                        default=1)

    parser.add_argument("--num_trials",
                        help="Bayesian audits work by simulating the ballots "
                             "that haven't been sampled to estimate the "
                             "chance that each choice would win a full "
                             "hand recount. This argument specifies how "
                             "many trials are used to compute these "
                             "estimates.",
                        type=int,
                        default=25000)

    parser.add_argument("--pseudocount_base",
                        help="The pseudocount value used for reported-choice/"
                        "actual-choice pairs that are unequal.  The default "
                        "value is 1, a relatively small value, indicating an "
                        "expectation that scanner errors are rare.",
                        type=int,
                        default=1)

    parser.add_argument("--pseudocount_match",
                        help="The pseudocount value used for reported-choice/"
                        "actual-choice pairs that are equal.  The default "
                        "value is 50, a relatively large value, indicating an "
                        "expectation that scanner is generally accurate.",
                        type=int,
                        default=50)

    parser.add_argument("--n_winners",
                        help="The parameter n_winners determines how many "
                        "winners there are in a contest.  The top n_winners "
                        "vote getters win the contest.",
                        type=int,
                        default=1)

    parser.add_argument("--v",
                        help="Verbose output.",
                        action='store_true')

    args = parser.parse_args()

    if args.collections_file is None:
        parser.print_help()
        sys.exit()

    return args


def read_csv(path_to_csv, expected_fieldnames):
    """
    Read CSV file and return list of dicts representing rows.

    Fieldnames have leading/trailing blanks removed, and are
    lower-cased.
    Values are converted to type "int" whenever possible.
    Checks that field names are exactly as expected.

    """

    with open(path_to_csv) as csv_file:

        reader = csv.DictReader(csv_file)
        # reader.fieldnames are headers in order
        # remove leading/trailing blanks from fieldnames
        # make lower case
        reader.fieldnames = [x.strip().lower() for x in reader.fieldnames]

        if set(reader.fieldnames) != set(expected_fieldnames):
            print("Error: in file {}, the fieldnames found \n"
                  "  {}\n"
                  "are not equal to those expected:\n"
                  "  {}"
                  .format(path_to_csv, reader.fieldnames, expected_fieldnames))
            sys.exit()

        rows = []
        for row in reader:
            new_row = dict()
            for name in reader.fieldnames:
                if row[name] is None:
                    val = ""
                else:
                    val = row[name].strip()
                val = convert_to_int_if_possible(val)
                new_name = name.strip().lower()
                new_row[new_name] = val
            rows.append(new_row)

        return rows


def read_and_process_collections(path_to_collections_file):
    """
    Return list of collection names and dict mapping names to sizes.

    Input:

        path_to_collections_file     -- (string) path to collections.csv file

    Output:

        collection_names             -- list of all collection names

        collection_size              -- dict mapping collection names to sizes

    """

    collections_rows = read_csv(path_to_collections_file,
                                ['collection', 'votes', 'comment'])

    collection_names = list([row['collection'] for row in collections_rows])

    collection_size = {row['collection']: row['votes']
                       for row in collections_rows}

    # Check that all collection names are distinct
    dupes = duplicates(collection_names)
    if dupes != []:
        raise ValueError("Repeated collection names in {}: {}"
                         .format(path_to_collections_file, dupes))

    return collection_names, collection_size, collections_rows


def read_and_process_reported(path_to_reported_file,
                              collection_names,
                              collection_size,
                              path_to_collections_file):
    """
    Read and process file for reported vote totals.

    Input:

        path_to_reported_file    -- (string) path to reported.csv file
        collection_names         -- list of all collection names
        collection_size          -- mapping: collection_name ->
                                             size (number of votes cast)
        path_to_collections_file -- (string) path to collections.csv
                                    file

    Returns:

        reported_choices        -- list of reported choices, over all
                                   collections, sorted into order
                                   alphabetically
        reported_size           -- mapping collection ->
                                   reported choice-> votes
        reported_rows           -- list of dicts, one per row
    """

    reported_rows = read_csv(path_to_reported_file,
                             ['collection', 'reported', 'votes',
                              'comment'])

    # Collect all reported choices
    reported_choices = list()
    for row in reported_rows:
        reported_choice = row['reported']
        if reported_choice not in reported_choices:
            reported_choices.append(reported_choice)
    reported_choices = sorted(reported_choices)

    # Initialize reported_size
    # for all collections, reported choices, set to zero
    reported_size = dict()
    for collection_name in collection_names:
        reported_size[collection_name] = dict()
        for reported_choice in reported_choices:
            reported_size[collection_name][reported_choice] = 0

    # Save votes for reported choice for each collection
    for row in reported_rows:
        collection_name = row['collection']
        if collection_name not in reported_size:
            print("ERROR: {} has illegal collection name: {} (IGNORED)"
                  .format(path_to_reported_file, collection_name))
            continue
        reported_choice = row['reported']
        reported_size[collection_name][reported_choice] = row['votes']

    # Check that total number of reported votes in each collection is
    # equal to sum over reported choices of number of votes for that
    # reported choice
    for collection_name in collection_names:
        n_reported = sum([reported_size[collection_name][reported_choice]
                          for reported_choice in
                          reported_size[collection_name]])
        if n_reported != collection_size[collection_name]:
            print("Collection '{}' has inconsistent reported vote counts:\n"
                  "  {} over all reported choices (from {}), but\n"
                  "  {} total reported votes cast (from {})."
                  .format(collection_name,
                          n_reported,
                          path_to_reported_file,
                          collection_size[collection_name],
                          path_to_collections_file))
            sys.exit()

    return reported_choices, reported_size, reported_rows


def read_and_process_sample(path_to_sample_file,
                            collection_names,
                            reported_choices):
    """
    Read sample.csv file and process it.

    Input:

        path_to_sample_file         -- (string) path to input sample.csv

        collection_names            -- list of collection names

        reported_choices            -- list of reported choices
                                       (from reported.csv)

    Return:

        actual_choices              --  list of all actual choices,
                                        plus all reported choices
                                        (ones that don't start with
                                        an initial "-"),
                                        sorted into in alphabetical order

        sample_dict                 --  nested mapping of dicts:
                                        collection_name -> reported_choice ->
                                            actual_choice -> votes
    """

    sample_rows = read_csv(path_to_sample_file,
                           ['collection', 'reported', 'actual', 'votes',
                            'comment'])

    # collect all actual choices (including reported choices)
    actual_choices = [choice for choice in reported_choices
                      if choice[0] != '-']
    for row in sample_rows:
        actual_choice = row['actual']
        if actual_choice not in actual_choices:
            actual_choices.append(actual_choice)
    actual_choices = sorted(actual_choices)

    # Initialize sample dict
    # for all possible collection, reported_choice, actual choice triples
    # set counts to zero
    sample_dict = dict()
    for collection in collection_names:
        sample_dict[collection] = dict()
        for reported_choice in reported_choices:
            sample_dict[collection][reported_choice] = dict()
            for actual_choice in actual_choices:
                sample_dict[collection][reported_choice][actual_choice] = 0

    for row in sample_rows:
        collection_name = row['collection']
        assert collection_name in sample_dict
        collection_dict = sample_dict[collection_name]
        reported_choice = row['reported']
        assert reported_choice in collection_dict
        reported_dict = collection_dict[reported_choice]
        actual_choice = row['actual']
        reported_dict[actual_choice] += row['votes']  # add, not replace

    return actual_choices, sample_dict, sample_rows


#
# Main computational routines
#

def dirichlet_multinomial(
        stratum_sample_tally,
        stratum_pseudocounts,
        stratum_size,
        rs):
    """
    Return a stratum tally according to the Dirichlet multinomial
    distribution, given a stratum_sample_tally, the number of votes cast
    in the stratum, and a random state.  The tally includes the
    stratum_sample_tally, so the returned tally sums to stratum_size.
    This routine is equivalent to the "Restore" procedure described
    in the "Bayesian Election Audits in One Page" note cited above.

    Input Parameters:

         stratum_sample_tally  -- a list of integers, where the i'th element
                                  is the number of votes that choice i received
                                  in the sample.
                                  (Here i indexes into actual_choices.)

         stratum_pseudocounts -- a list of integers of exactly the same length
                                 as stratum_sample_tally, giving as the i-th
                                 element the Bayesian pseudocount for the i-th
                                 choices, as a way of defining the prior.

         stratum size         -- an integer equal to the number of
                                 votes cast in this contest in this stratum.

         rs                   -- a Numpy RandomState object that is used for
                                 any random functions in the simulation of
                                 the nonsample votes. In particular,
                                 the gamma functions are made deterministic
                                 using this state.

    Returns:

        multinomial_sample       -- a list of integers, which sums up
                                    to (stratum_size - stratum_sample_size).
                                    The i'th element is equal to the
                                    simulated number of votes for choice i in
                                    the remaining, unsampled votes.

    """

    stratum_sample_size = sum(stratum_sample_tally)

    if stratum_sample_size > stratum_size:
        raise ValueError("stratum_size {} less than stratum_sample_size {}."
                         .format(stratum_size, stratum_sample_size))

    nonsample_size = stratum_size - stratum_sample_size

    stratum_sample_tally_plus_pseudocounts = \
        stratum_sample_tally + stratum_pseudocounts

    gamma_sample = [rs.gamma(k)
                    for k in stratum_sample_tally_plus_pseudocounts]
    gamma_sample_sum = float(sum(gamma_sample))
    gamma_sample = [k / gamma_sample_sum for k in gamma_sample]

    multinomial_sample_tally = rs.multinomial(nonsample_size, gamma_sample)

    # print("stratum_sample_tally:", stratum_sample_tally)
    # print("stratum_pseudocounts:", stratum_pseudocounts)
    # print(multinomial_sample_tally)
    return multinomial_sample_tally + stratum_sample_tally


def generate_restored_sample_tally(stratum_sample_tally,
                                   stratum_pseudocounts,
                                   stratum_size,
                                   seed):
    """
    Given a stratum_sample_tally, a stratum_prior (expressed as pseudocounts),
    the stratum_size, and a seed,  generate a restored_sample_tally in the
    contest using the Dirichlet multinomial distribution.

    Input Parameters:

         stratum_sample_tally  -- a list of integers, where the i'th index in
                                  sample_tally corresponds to the number of
                                  votes that choice i received in the sample.
                                  (Here i indexes into actual_choices.)

         stratum_pseudocounts -- a list of integers of exactly the same length
                                 as stratum_sample_tally, giving as the i-th
                                 element the Bayesian pseudocount for the i-th
                                 choices, as a way of defining the prior.

         stratum size         -- an integer equal to the number of
                                 votes cast in this contest in this stratum.

         seed                 -- an integer or None. Assuming that it isn't
                                 None, we use it to seed the random state
                                 for the audit.

    Returns:

        restored_sample_tally -- a list of integers, which sums up
                                 to  total_num_votes.
                                 The i'th index represents the simulated
                                 number of votes for choice i in the
                                 "restored sample".
    """

    rs = create_rs(seed)
    restored_sample_tally = \
        dirichlet_multinomial(stratum_sample_tally,
                              stratum_pseudocounts,
                              stratum_size,
                              rs)
    return restored_sample_tally


def compute_winner(strata_sample_tallies,
                   strata_pseudocounts,
                   total_num_votes,
                   seed,
                   actual_choices,
                   n_winners,
                   pretty_print=False):
    """
    Given a list of sample tallies (one sample tally per collection)
    a list giving the total number of votes cast in each collection,
    and a random seed (an integer)
    compute the winner in a single simulation.
    For each collection, we use the Dirichlet-Multinomial distribution to
    generate a restored_sample_tally. Then, we sum over all the collections
    to produce our final tally and calculate the predicted winner(s) over
    all the collections in the contest.

    Input Parameters:

        strata_sample_tallies    -- a list of lists. Each list is
                                    a stratum sample tally.
                                    So, strata_sample_tallies[i]
                                    represents the stratum tally for stratum i.
                                    Then, strata_sample_tallies[i][j]
                                    represents the number of votes choice j
                                    receives in stratum i.

        strata_pseudocounts      -- a list of lists, same shape as
                                    strata_sample_tallies.
                                    i-th element gives pseudocounts
                                    for i-th stratum.

        total_num_votes          -- a list whose i-th element is an integer
                                    equal to the number of ballots that were
                                    cast in this contest in the i-th stratum.

        seed                     -- an integer or None. Assuming that it
                                    isn't None, we use it to seed the random
                                    state for the audit.

        n_winners                -- an integer, parsed from command-line args.
                                    Its default value is 1, which means we only
                                    calculate a single winner for the contest.
                                    For other values n_winners, we simulate
                                    the unnsampled votes and define a win for
                                    choice i as any time they are in the top
                                    n_winners choices in the final tally.

        pretty_print             -- a Boolean, which defaults to False.
                                    When True, we print the winning choice,
                                    the number of votes it received and the
                                    final vote tally for all the choices.

    Returns:

        winner                   -- an integer, representing the index of the
                                    choice that won the contest.
    """

    final_tallies = np.zeros(len(actual_choices))
    for i in range(len(strata_sample_tallies)):
        stratum_sample_tally = strata_sample_tallies[i]
        stratum_pseudocounts = strata_pseudocounts[i]
        stratum_size = total_num_votes[i]
        restored_sample_tally = generate_restored_sample_tally(
            stratum_sample_tally,
            stratum_pseudocounts,
            stratum_size,
            seed)
        final_collection_tally = restored_sample_tally
        final_tallies = final_tallies + final_collection_tally
    final_tallies = [(k, final_tallies[k]) for k in range(len(final_tallies))]
    final_tallies.sort(key=lambda x: x[1])
    winners_with_tallies = final_tallies[-n_winners:]
    winners = [winner_tally[0] for winner_tally in winners_with_tallies]

    if pretty_print:
        results_str = ''
        for i in range(len(winners)):
            results_str += (
                "Choice {} is a winner with {} votes. "
                .format(winners[i], final_tallies[winners[i]][1]))
        results_str += (
            "The final vote tally for all the choices was {}"
            .format([final_tally[1] for final_tally in final_tallies]))
        print(results_str)

    return winners


def compute_win_probs(strata_sample_tallies,
                      strata_pseudocounts,
                      total_num_votes,
                      seed,
                      num_trials,
                      actual_choices,
                      n_winners):
    """
    Run num_trials simulations of the Bayesian audit to estimate
    the probability that each candidate would win a full recount.

    In particular, we run a single iteration of a Bayesian audit
    (extend each collection's sample to simulate all the votes in the
    collection and calculate the overall winner across counties)
    num_trials times.

    Input Parameters:

        strata_sample_tallies  -- a list of lists. Each list represents
                                  the sample tally for a given collection.
                                  So, sample_tallies[i] represents the
                                  tally for collection i. Then,
                                  sample_tallies[i][j] represents the
                                  number of votes choice j receives in
                                  collection i.

        strata_pseudocounts    -- an array of the same shape as
                                  strata_sample_tallies;
                                  the i-th element of this array gives
                                  the Bayesian prior in the form of
                                  pseudocounts.

        total_num_votes        -- an integer representing the number of
                                  ballots that were cast in this contest

        seed                   -- an integer or None. Assuming that it isn't
                                  None, we use it to seed the random state
                                  for the audit.

        num_trials             -- an integer which represents how many
                                  simulations of the Bayesian audit trial
                                  we run to estimate the win probabilities
                                  of the choices.

        actual_choices         -- an ordered list of strings, containing
                                  the name of every choice in the contest
                                  being audited.

        n_winners              -- an integer, parsed from the command-line
                                  args. Its default value is 1, which means
                                  we only calculate a single winner for the
                                  contest.
                                  For other values of n_winners, we simulate
                                  the unnsampled votes and define a win for
                                  choice i as any time they are in the top
                                  n_winners choices in the final tally.

    Returns:

        win_probs              -- a list of pairs (i, p) where p is the
                                  fractional representation of the number
                                  of trials that candidate i has won
                                  out of the num_trials simulations.
    """

    win_count = np.zeros(1+len(actual_choices))
    for i in range(num_trials):
        # We want a different seed per trial.
        # Adding i to seed caused correlations, as numpy apparently
        # adds one per trial, so we multiply i by 314...
        seed_i = seed + i*314159265
        winners = compute_winner(strata_sample_tallies,
                                 strata_pseudocounts,
                                 total_num_votes,
                                 seed_i,
                                 actual_choices,
                                 n_winners)
        for winner in winners:
            win_count[winner+1] += 1
    win_probs = [(i, win_count[i]/float(num_trials))
                 for i in range(1, len(win_count))]
    return win_probs


#
# Output routines
#

def print_results(actual_choices, win_probs, n_winners):
    """
    Given list of all choices and win_probs pairs, print summary
    of the Bayesian audit simulations.

    Input Parameters:

        actual_choices              -- an ordered list of strings,
                                       containing the name of every choice
                                       reported or seen in the contest
                                       we are auditing.

        win_probs                   -- a list of pairs (i, p) where p is
                                       the fractional representation of the
                                       number of trials that choice i has
                                       won out of the num_trials simulations.

        n_winners                   -- an integer, parsed from the command
                                       line. Its default value is 1, which
                                       means we only calculate a single
                                       winner for the contest.
                                       For other values of n_winners, we
                                       simulate the unnsampled votes and
                                       define a win for choice i as any trial
                                       it is in the top n_winners choices in
                                       the final tally.

    Returns:

        None                        -- but prints a summary of how often each
                                       choice has won in the simulations.
    """

    print("BCTOOL (Bayesian ballot-comparison tool version 0.8)")

    want_sorted_results = True
    if want_sorted_results:
        sorted_win_probs = sorted(
            win_probs, key=lambda tup: tup[1], reverse=True)
    else:
        sorted_win_probs = win_probs

    if n_winners == 1:
        print("{:<24s} \t {:<s}"
              .format("Choice        ",
                      "Estimated probability of winning a full recount"))
    else:
        print("{:<24s} \t {:<s} {} {:<s}"
              .format("Choice        ",
                      "Estimated probability of being among the top ",
                      n_winners,
                      "winners in a full recount"))

    for choice_index, prob in sorted_win_probs:
        choice_name = str(actual_choices[choice_index - 1])
        print(" {:<24s} \t  {:6.2f} %  "
              .format(choice_name, 100*prob))


def main():
    """
    Parse command-line arguments, compute and print answers.
    """

    args = parse_args()

    if args.v:
        print("Reading collections file: ", args.collections_file)

    collection_names, collection_size, collections_rows = \
        read_and_process_collections(args.collections_file)

    if args.v:
        print("Collection names: ", collection_names)
        for row in collections_rows:
            print("    ", row)

    if args.v:
        print("Reading reported choice tally file: ", args.reported_file)

    reported_choices, reported_size, reported_rows = \
        read_and_process_reported(args.reported_file,
                                  collection_names,
                                  collection_size,
                                  args.collections_file)

    if args.v:
        print("Reported choices: ", reported_choices)
        for row in reported_rows:
            print("    ", row)

    if args.v:
        print("Reading sample audit data file: ", args.sample_file)

    actual_choices, sample_dict, sample_rows = \
        read_and_process_sample(args.sample_file,
                                collection_names,
                                reported_choices)

    if args.v:
        print("All reported choices and choices seen in audit: ",
              actual_choices)
        for row in sample_rows:
            print("    ", row)

    # Stratify by (collection, choice) pairs
    strata = []                 # list of (collection, reported_choice) pairs
    strata_size = []            # corresponding list of # reported votes
    for collection in collection_names:
        for reported_choice in reported_choices:
            stratum = (collection, reported_choice)
            strata.append(stratum)
            # Record stratum size (number of votes reported cast)
            strata_size.append(reported_size[collection][reported_choice])

    # create sample tallies for each strata
    strata_sample_tallies = []
    strata_pseudocounts = []
    for (collection, reported_choice) in strata:
        stratum_sample_tally = []
        stratum_pseudocounts = []
        for actual_choice in actual_choices:
            count = sample_dict[collection][reported_choice][actual_choice]
            stratum_sample_tally.append(count)
            if reported_choice == actual_choice:
                stratum_pseudocounts.append(args.pseudocount_match)
            else:
                stratum_pseudocounts.append(args.pseudocount_base)
        strata_sample_tallies.append(np.array(stratum_sample_tally))
        strata_pseudocounts.append(np.array(stratum_pseudocounts))
        # print(collection, reported_choice,
        #       stratum_sample_tally, stratum_pseudocounts)

    win_probs = compute_win_probs(
                    strata_sample_tallies,
                    strata_pseudocounts,
                    strata_size,
                    args.audit_seed,
                    args.num_trials,
                    actual_choices,
                    args.n_winners)

    print_results(actual_choices, win_probs, args.n_winners)


if __name__ == '__main__':

    main()
