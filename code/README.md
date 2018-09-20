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

[Optional:] MODEL.CSV format:

    Example:

        Collection,  Reported,  Actual,    Percentage,       Comment
        Bronx,       Yes,       Yes,         95
        Bronx,       Yes,       No,          5
        Bronx,       No,        No,          100
        Queens,      Yes,       Yes,         100
        Queens,      No,        No,          100
        Mail-In,     Yes,       Yes,         100      
        Mail-In,     No,        No,          100

    The model.csv file is used for work estimation. When we want to estimate
    how long a Bayesian audit will take to complete, we can define a model
    of what the rest of the votes will look like. In this case, we claim that
    out of the reported votes for Yes, in Bronx, 95% of them will correspond
    to an actual vote of Yes. 5% of the reported votes of Yes in Bronx will
    correspond to an actual vote of No - this can be used to model human estimates
    of how often a scanner, in the Bronx will make a mistake. In this example,
    we assume that all the other possible pairings occur exactly as reported.

    Using this model.csv file, we can estimate how much work remains in the
    audit. We can scale up each reported votes stratum, according to its
    sample size. Then, inside each stratum, we can model each (reported, actual)
    vote pair, according to our defined model. We can calculate how much we
    will need to increase our sample size by for our Bayesian risk limit
    to be satisfied.

    We can use the work estimation module, by passing in the flag --estimate_work=True,
    and the path to the model file.

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

