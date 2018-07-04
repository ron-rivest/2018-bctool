# bctool.py
# Authors: Ronald L. Rivest, Mayuri Sridhar, Zara A Perumal
# June 26, 2018
# python3
# Comment: derived from bptool.py

"""

This module provides routines for computing the winning probabilities
for various candidates, given audit sample data, using a Bayesian
model, in a ballot-comparison audit of a plurality election.  The
election may be single-jurisdiction or multi-jurisdiction.  In this
module we call a jurisdiction a "collection".


Like audit-lab, the Bayesian model defaults to using a prior pseudocount of 
    "+1"   for each matrix tally cell entry (R, A) if R != A
    "+50"  for each matrix tally cell entry (R, A) if R == A
for each pair of candidates R, A (reported, actual).


This module can be used for ballot-polling audits too, if all
reported votes are set to "MISSING" (or some other value that
doesn't otherwise occur as an actual candidate name).


If this module is imported, rather than used stand-alone, then the procedure
    compute_win_probs
can compute the desired probability of each candidate winning a full recount,
given sample tallies for each collection.


For command-line usage, there are really just one modes:

To use this program, give a command like

        python3 bctool.py collections.csv reported.csv sample.csv

    where 

        python3            is the python3 interpreter on your system 
                           (this might be just called "python")
                           Besides being python3, your installation must include
                           the numpy package.  Anaconda provides an excellent
                           installation framework.

        collections.csv    is the name of a file giving the number of (cast) votes 
                           in each collection

        reported.csv       is the name of a file giving the reported number of 
                           votes for each candidate in each collection

        sample.csv         is the name of a file giving the number of 
                             (reported vote, actual vote)
                           pairs in the sample for each collection.  Here the 
                           reported vote is the candidate that the scanner reported
                           to have been on the ballot; the actual vote is the 
                           candidate revealed by a manual examination of the ballot.
                            

File formats are as follows; all are CSV files with one header line, then data lines.

COLLECTIONS.CSV format:

    Example:

        Collection,      Votes
        Bronx,           11000
        Queens,         120000
        Mail-In,         56000

    with one header line, as shown, 
    then one data line for each collection of paper ballots 
    that may be sampled; each data line gives the name of 
    the collection and then the number of votes in that
    collection (that is, the number of paper ballots in the
    collection).


REPORTED.CSV format

    Example:

        Collection,  Reported,    Votes
        Bronx,       Yes,          8500    
        Bronx,       No,           2500
        Bronx,       Overvote,        0
        Queens,      Yes,        100000
        Queens,      No,          20000
        Queens,      Overvote,        0
        Mail-In,     Yes,          5990
        Mail-In,     No,          50000
        Mail-In,     Overvote,       10

    with one header line, as shown,
    then one data line for each collection and each possible reported vote
    in that collection, giving the number of such reported votes.
    A possible reported vote must be listed, even if it is not reported
    to have occurred.
    For each collection, the sum of the Votes given should sum to the Votes
    value given in the corresponding line in the collections.csv file.
    Each collection must have the same list of possible reported votes
    (even if some of them have zero votes reported).


SAMPLE.CSV format:

    Example:

        Collection,  Reported,  Actual,    Votes
        Bronx,       Yes,       Yes,         62
        Bronx,       No,        No,          21
        Bronx,       No,        Overvote,     1
        Queens,      Yes,       Yes,        504
        Queens,      No,        No,          99
        Mail-In,     Yes,       Yes,         17       
        Mail-In,     No,        No,          73

    with one header line, as shown
    then one data line for each collection for each reported/actual vote
    pair that was seen in the audit sample at least once, giving the 
    number of times it was seen in the sample.  There is no need to 
    give a data line for a reported/actual pair that hasn't been seen
    in the sample.  The Reported names must be names that appeared in
    the reported.csv file; the actual names may be anything (thus allowing
    write-ins, for example).

    As the audit progresses, the file sample.csv will change, but the
    other two files should not.  Only the file sample.csv has data 
    obtained from manually auditing a random sample of the paper ballots.

    It is assumed here that sampling is done without replacement.

    This module does not assume that each collection is sampled at the same
    rate --- the audit might look at 1% of the ballots in the Bronx
    collection while looking at 2% of the ballots in the Queens collection.  
    A Bayesian audit can easily work with such differences in sampling rates.

    (The actual sampling rates are presumed to be determined by other
    modules; the present module only computes winning probabilities based
    on the sample data obtained so far.)


There are optional parameters as well, to see the documentation of them, do
    python bctool.py --h

More description of Bayesian auditing methods can be found in:

    A Bayesian Method for Auditing Elections
    by Ronald L. Rivest and Emily Shen
    EVN/WOTE'12 Proceedings
    http://people.csail.mit.edu/rivest/pubs.html#RS12z

    Bayesian Tabulation Audits: Explained and Extended
    by Ronald L. Rivest 
    2018
    http://people.csail.mit.edu/rivest/pubs.html#Riv18a    

    Bayesian Election Audits in One Page
    by Ronald L. Rivest
    2018
    http://people.csail.mit.edu/rivest/pubs.html#Riv18b    

"""

import argparse

from copy import deepcopy
import csv
import sys

import numpy as np

##############################################################################
## Some global variables
##############################################################################

# Bayes prior hyperparameters; constants for now; allow CLI setting later
PSEUDOCOUNT_BASE = 1
PSEUDOCOUNT_MATCH = 50

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


##############################################################################
## Main computational routines
##############################################################################

def dirichlet_multinomial(
    sample_tally, total_num_votes, rs, pseudocount_for_prior):
    """
    Return a sample according to the Dirichlet multinomial distribution,
    given a sample tally, the number of votes in the election,
    and a random state. There is an additional pseudocount of
    one vote per candidate in this simulation.

    Input Parameters:

    -sample_tally is a list of integers, where the i'th index
    in sample_tally corresponds to the number of votes that candidate
    i received in the sample.

    -total_num_votes is an integer representing the number of
    ballots that were cast in this election.

    -rs is a Numpy RandomState object that is used for any
    random functions in the simulation of the remaining votes. In particular,
    the gamma functions are made deterministic using this state.

    -pseudocount_for_prior is an integer, defining how many votes we add
    in by default, for each candidate, as a prior. The default of 1 gives
    a uniform prior over all the candidates that they each automatically
    receive 1 vote.

    Returns:

    -multinomial_sample is a list of integers, which sums up
    to the total_num_votes - sample_size. The i'th index represents the
    simulated number of votes for candidate i in the remaining, unsampled
    votes.
    """

    sample_size = sum(sample_tally)
    if sample_size > total_num_votes:
        raise ValueError("total_num_votes {} less than sample_size {}."
                         .format(total_num_votes, sample_size))

    nonsample_size = total_num_votes - sample_size

    sample_with_prior = deepcopy(sample_tally)
    sample_with_prior = [k + pseudocount_for_prior
                         for k in sample_with_prior]

    gamma_sample = [rs.gamma(k) for k in sample_with_prior]
    gamma_sample_sum = float(sum(gamma_sample))
    gamma_sample = [k / gamma_sample_sum for k in gamma_sample]

    multinomial_sample = rs.multinomial(nonsample_size, gamma_sample)

    return multinomial_sample


def generate_nonsample_tally(
    sample_tally, total_num_votes, seed, pseudocount_for_prior):
    """
    Given a sample_tally, the total number of votes in an election, and a seed,
    generate the nonsample tally in the election using the Dirichlet multinomial
    distribution.

    Input Parameters:

    -sample_tally is a list of integers, where the i'th index
    in sample_tally corresponds to the number of votes that candidate
    i received in the sample.

    -total_num_votes is an integer representing the number of
    ballots that were cast in this election.

    -seed is an integer or None. Assuming that it isn't None, we
    use it to seed the random state for the audit.

    -pseudocount_for_prior is an integer, defining how many votes we add
    in by default, for each candidate, as a prior. The default of 1 gives
    a uniform prior over all the candidates that they each automatically
    receive 1 vote.

    Returns:

    -nonsample_tally is list of integers, which sums up
    to the total_num_votes - sample_size. The i'th index represents the
    simulated number of votes for candidate i in the remaining, unsampled
    votes.
    """

    rs = create_rs(seed)
    nonsample_tally = dirichlet_multinomial(
        sample_tally, total_num_votes, rs, pseudocount_for_prior)
    return nonsample_tally


def compute_winner(sample_tallies, total_num_votes, vote_for_n,
                   seed, pseudocount_for_prior, pretty_print=False):
    """
    Given a list of sample tallies (one sample tally per collection)
    a list giving the total number of votes cast in each collection,
    and a random seed (an integer)
    compute the winner in a single simulation.
    For each collection, we use the Dirichlet-Multinomial distribution to generate
    a nonsample tally. Then, we sum over all the counties to produce our
    final tally and calculate the predicted winner over all the counties in
    the election.

    Input Parameters:

    -sample_tallies is a list of lists. Each list represents the sample tally
    for a given collection. So, sample_tallies[i] represents the tally for collection
    i. Then, sample_tallies[i][j] represents the number of votes candidate
    j receives in collection i.

    -total_num_votes is an integer representing the number of
    ballots that were cast in this election.

    -seed is an integer or None. Assuming that it isn't None, we
    use it to seed the random state for the audit.

    -vote_for_n is an integer, parsed from the command-line args. Its default
    value is 1, which means we only calculate a single winner for the election.
    For other values n, we simulate the unnsampled votes and define a win
    for candidate i as any time they are in the top n candidates in the final
    tally.

    -pseudocount_for_prior is an integer, defining how many votes we add
    in by default, for each candidate, as a prior. The default of 1 gives
    a uniform prior over all the candidates that they each automatically
    receive 1 vote.

    -pretty_print is a Boolean, which defaults to False. When it's set to
    True, we print the winning candidate, the number of votes they have
    received and the final vote tally for all the candidates.

    Returns:

    -winner is an integer, representing the index of the candidate who
    won the election.
    """

    final_tallies = None
    for i, sample_tally in enumerate(sample_tallies):   # loop over collections
        nonsample_tally = generate_nonsample_tally(
            sample_tally, total_num_votes[i], seed, pseudocount_for_prior)
        final_collection_tally = [sum(k)
                              for k in zip(sample_tally, nonsample_tally)]
        if final_tallies is None:
            final_tallies = final_collection_tally
        else:
            final_tallies = [sum(k)
                           for k in zip(final_tallies, final_collection_tally)]
    final_tallies = [(k, final_tallies[k]) for k in range(len(final_tallies))]
    final_tallies.sort(key = lambda x: x[1])
    winners_with_tallies = final_tallies[-vote_for_n:]
    winners = [winner_tally[0] for winner_tally in winners_with_tallies]
    if pretty_print:
        results_str = ''
        for i in range(len(winners)):
            results_str += (
                "Candidate {} is a winner with {} votes. ".format(
                winners[i], final_tallies[winners[i]][1]))
        results_str += (
            "The final vote tally for all the candidates "
            "was {}".format(
                [final_tally[1] for final_tally in final_tallies]))
        print(results_str)
    return winners


def compute_win_probs(sample_tallies,
                      collection_total_votes,
                      seed,
                      num_trials,
                      candidate_names,
                      vote_for_n):
    """

    Runs num_trials simulations of the Bayesian audit to estimate
    the probability that each candidate would win a full recount.

    In particular, we run a single iteration of a Bayesian audit
    (extend each collection's sample to simulate all the votes in the
    collection and calculate the overall winner across counties)
    num_trials times.

    Input Parameters:

    -sample_tallies is a list of lists. Each list represents the sample tally
    for a given collection. So, sample_tallies[i] represents the tally for collection
    i. Then, sample_tallies[i][j] represents the number of votes candidate
    j receives in collection i.

    -total_num_votes is an integer representing the number of
    ballots that were cast in this election.

    -seed is an integer or None. Assuming that it isn't None, we
    use it to seed the random state for the audit.

    -num_trials is an integer which represents how many simulations
    of the Bayesian audit we run, to estimate the win probabilities
    of the candidates.

    -candidate_names is an ordered list of strings, containing the name of
    every candidate in the contest we are auditing.

    -vote_for_n is an integer, parsed from the command-line args. Its default
    value is 1, which means we only calculate a single winner for the election.
    For other values n, we simulate the unnsampled votes and define a win
    for candidate i as any time they are in the top n candidates in the final
    tally.

    Returns:

    -win_probs is a list of pairs (i, p) where p is the fractional
    representation of the number of trials that candidate i has won
    out of the num_trials simulations.
    """

    num_candidates = len(candidate_names)
    win_count = [0]*(1+num_candidates)
    for i in range(num_trials):
        # We want a different seed per trial.
        # Adding i to seed caused correlations, as numpy apparently
        # adds one per trial, so we multiply i by 314...
        seed_i = seed + i*314159265
        winners = compute_winner(sample_tallies,
                                total_num_votes,
                                vote_for_n,
                                seed_i)
        for winner in winners:
            win_count[winner+1] += 1
    win_probs = [(i, win_count[i]/float(num_trials))
                 for i in range(1, len(win_count))]
    return win_probs


##############################################################################
## Routines for command-line interface and file (csv) input
##############################################################################

def print_results(candidate_names, win_probs, vote_for_n):
    """
    Given list of candidate_names and win_probs pairs, print summary
    of the Bayesian audit simulations.

    Input Parameters:

    -candidate_names is an ordered list of strings, containing the name of
    every candidate in the contest we are auditing.

    -win_probs is a list of pairs (i, p) where p is the fractional
    representation of the number of trials that candidate i has won
    out of the num_trials simulations.

    -vote_for_n is an integer, parsed from the command-line args. Its default
    value is 1, which means we only calculate a single winner for the election.
    For other values n, we simulate the unnsampled votes and define a win
    for candidate i as any time they are in the top n candidates in the final
    tally.

    Returns:

    -None, but prints a summary of how often each candidate has won in
    the simulations.
    """

    print("BCTOOL (Bayesian ballot-comparison tool version 0.8)")

    want_sorted_results = True
    if want_sorted_results:
        sorted_win_probs = sorted(
            win_probs, key=lambda tup: tup[1], reverse=True)
    else:
        sorted_win_probs = win_probs

    if vote_for_n == 1:
        print("{:<24s} \t {:<s}"
              .format("Candidate name",
                      "Estimated probability of winning a full recount"))
    else:
        print("{:<24s} \t {:<s} {} {:<s}"
              .format("Candidate name",
                      "Estimated probability of being among the top",
                      vote_for_n,
                      "winners in a full recount"))

    for candidate_index, prob in sorted_win_probs:
        candidate_name = str(candidate_names[candidate_index - 1])
        print(" {:<24s} \t  {:6.2f} %  "
              .format(candidate_name, 100*prob))


def convert_to_int_if_possible(x):
    """
    Convert x to int if possible and return it; else just return x.
    """

    try:
        x = int(x)
        return x
    except:
        return x


def read_csv(path_to_csv, expected_fieldnames):
    """
    Read CSV file and return dict representation of it.
    Check that field names are exactly as expected.

    """

    printing_wanted = True
    if printing_wanted:
        print()
        print("Reading csv file: ", path_to_csv)

    with open(path_to_csv) as csv_file:
    
        reader = csv.DictReader(csv_file)
        # reader.fieldnames are headers in order
        # remove leading/trailing blanks from fieldnames; lower case
        reader.fieldnames = [x.strip().lower() for x in reader.fieldnames]
        assert set(reader.fieldnames) == set(expected_fieldnames)

        rows = []
        for row in reader:        
            new_row = dict()
            for name in reader.fieldnames:
                new_name = name.strip().lower()
                val = row[name].strip()
                val = convert_to_int_if_possible(val)
                new_row[new_name] = val
            if printing_wanted:
                print(new_row)
            rows.append(new_row)

        return rows


def main():
    """
    Parse command-line arguments, compute and print answers.
    """

    parser = argparse.ArgumentParser(description=\
                                     'Bayesian Comparison Audit Process For'
                                     'A Single Contest '
                                     'Across One or More Counties')

    parser.add_argument("collections_file",
                        help="name of CSV file giving collection names and sizes.",
                        default = "collections.csv")

    parser.add_argument("reported_file",
                        help="name of CSV file giving number of times each candidate "
                             "was reportedly voted for in each collection.",
                        default = "reported.csv")

    parser.add_argument("sample_file",
                        help="name of CSV file giving number of times each "
                             "(reported vote, actual vote) pair was seen in "
                             "each collection in the audit sample.",
                        default = "sample.csv")

    parser.add_argument("--audit_seed",
                        help="For reproducibility, we provide the option to "
                             "seed the randomness in the audit. If the same "
                             "seed is provided, the audit will return the "
                             "same results.",
                        type=int,
                        default=1)

    parser.add_argument("--num_trials",
                        help="Bayesian audits work by simulating the data "
                             "which hasn't been sampled to estimate the "
                             "chance that each candidate would win a full "
                             "hand recount. This argument specifies how "
                             "many trials are done to compute these "
                             "estimates.",
                        type=int,
                        default=10000)

    parser.add_argument("--vote_for_n",
                        help="If we want to simulate voting for multiple "
                        "candidates at a time, we can use this parameter "
                        "to specify how many candidates a single voter "
                        "can vote for. The simulations will then count a candidate"
                        "as a winner, each time they appear in the top N "
                        "of the candidates, in the simulated elections.",
                        type=int,
                        default=1)

    args = parser.parse_args()
    if args.collections_file is None:
        parser.print_help()
        sys.exit()

    vote_for_n = args.vote_for_n

    collections_rows = read_csv(args.collections_file,
                                ['collection', 'votes'])
    reported_rows = read_csv(args.reported_file,
                             ['collection', 'reported', 'votes'])
    sample_rows = read_csv(args.sample_file,
                           ['collection', 'reported', 'actual', 'votes'])
    
    ## process collections, yielding
    ##   collection_names: list of all collection names
    ##   collection_size: dict mapping collection names to sizes
    collection_names = list([row['collection'] for row in collections_rows])
    collection_size = { row['collection']: row['votes']
                        for row in collections_rows 
                      }
    ### add check for all collection names being unique here

    ## process reported yielding 
    ##   reported_names: set of reported names
    ##   reported_size: collections -> reported -> votes
    reported_names = set()
    for row in reported_rows:
        reported_names.add(row['reported'])
    print("reported_names:", reported_names)
    reported_size = {}
    for row in reported_rows:
        collection_name = row['collection']
        if collection_name not in reported_size:
            reported_size[collection_name] = dict()
        col_dict = reported_size[collection_name]
        reported_name = row['reported']
        col_dict[reported_name] = row['votes']
    # Check that every collection has every reported name
    for collection in reported_size:
        assert set(reported_size[collection])==set(reported_names), \
            "Not all collections have all reported names: {} {}" \
            .format(set(reported_size[collection]), set(reported_names))

    ## process sample_rows, yielding
    ##   actual_names = set of all actual names, plus all reported names
    ##   sample_dict:
    ##      collection_name -> reported_name -> actual_name -> votes
    actual_names = reported_names.copy()
    sample_dict = {}
    for row in sample_rows:
        collection_name = row['collection']
        if collection_name not in sample_dict:
            sample_dict[collection_name] = dict()
        col_dict = sample_dict[collection_name]
        reported_name = row['reported']
        if reported_name not in col_dict:
            col_dict[reported_name] = dict()
        rep_dict = col_dict[reported_name]
        actual_name = row['actual']
        rep_dict[actual_name] = row['votes']
        actual_names.add(actual_name)
    print("actual names:", actual_names)


    win_probs = compute_win_probs(\
                    candidate_names,                                 
                    sample_tallies,
                    collection_total_votes,
                    collection_sample_votes,                                  
                    args.audit_seed,
                    args.num_trials,
                    vote_for_n
                 )

    print_results(candidate_names, win_probs, vote_for_n)


if __name__ == '__main__':

    main()
