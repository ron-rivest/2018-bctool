"""
Simple website form for using BPTool online. Can be hosted
by running "python website.py"
"""
import csv
import cherrypy
import numpy as np
import os
import shutil
import pandas as pd

import bctool


class BCToolPage:

    @cherrypy.expose
    def index(self):
        # Ask for the parameters required for the Bayesian Audit.
        # Style parameters from 
        # https://www.w3schools.com/css/tryit.asp?filename=trycss_forms
        return '''
            <style>
            input[type=text], select {
                width: 50%;
                padding: 12px 20px;
                margin: 8px 0;
                display: inline-block;
                border: 1px solid #ccc;
                border-radius: 4px;
                box-sizing: border-box;
            }
            input[type=submit] {
                width: 100%;
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                padding: 15px 20px;
                margin: 8px 0;
                border-radius: 4px;
            }

            table, th, td {
                border: 1px solid black;
            }
            </style>
            <h1> Bayesian Ballot Comparison Tool </h1>
            <p>This module provides support for auditing of a single plurality contest
            over multiple jurisdictions using a Bayesian ballot-level
            comparison audit.</p>

            <p>This module provides routines for computing the winning probabilities
            for various choices, given audit sample data.  Thus, this program
            provides a risk-measuring functionality.</p>

            <p>More precisely, the code builds a Bayesian model of the unsampled
            ballots, given the sampled ballots.  This model is probabilistic,
            since there is uncertainty about what the unsampled ballots are.
            However, the model is generative: we generate many possible
            sets of likely unsampled ballots, and report the probability that
            various choices win the contest.  (See References below for
            more details.)</p>

            <p>The main output of the program is a report on the probability
            that each choice wins, in these simulations.</p>

            <p>The contest may be single-jurisdiction or multi-jurisdiction.
            More precisely, we assume that there are a number of "collections"
            of ballots that may be sampled.  Each relevant jurisdiction may
            have one or more such collections.  For example, a jurisdiction
            may be a county, with one collection for ballots submitted by
            mail, and one collection for ballots cast in-person.</p>

            <p>This module may be used for ballot-polling audits (where there
            no reported choices for ballots) or "hybrid" audits (where some
            collections have reported choices for ballots, and some have not).</p>

            <h2> References and Code</h2>
            <p> Descriptions of Bayesian auditing methods can be found in: </p>
            <ul>
            <li><a href="http://people.csail.mit.edu/rivest/pubs.html#RS12z">
            A Bayesian Method for Auditing Elections</a>
            by Ronald L. Rivest and Emily Shen
            EVN/WOTE'12 Proceedings
            </li>

            <li><a href="http://people.csail.mit.edu/rivest/pubs.html#Riv18a">
            Bayesian Tabulation Audits: Explained and Extended</a>
            by Ronald L. Rivest 
            2018
            </li> 

            <li><a href="http://people.csail.mit.edu/rivest/pubs.html#Riv18b">
            Bayesian Election Audits in One Page</a>
            by Ronald L. Rivest
            2018
            </li>

            </ul>

            <h2>Implementation Note</h2>
            <p> The code for this tool is available on github at 
            <a href="https://github.com/ron-rivest/2018-bctool">www.github.com/ron-rivest/2018-bctool</a>.
            This web form provides exactly the same functionality as the stand-alone 
            Python tool
            <a href=https://github.com/ron-rivest/2018-bctool>www.github.com/ron-rivest/2018-bctool/BCTool.py</a>. 
            The Python tool
            requires an environment set up with Python 3 and Numpy.
            This web form was implemented using 
            <a href="https://github.com/cherrypy/cherrypy">
            CherryPy
            </a>.
            <p>(See <a href="https://github.com/ron-rivest/2018-bptool">www.github.com/ron-rivest/2018-bptool</a>
            for similar code for a Bayesian ballot-level polling audit.  The code here was based
            in part on that code.)</p>
            </p>


            <form action="ComparisonAudit" method="post" enctype="multipart/form-data">

            <h2>Step 1: Upload Collections File</h2>

            <p> In the box below, upload a CSV file with the data about collections
            in the contest being audited. </p>

            <h3>
            COLLECTIONS.CSV format:</h3>

            <p>Example collections.csv file:</p>

            <table style="width:100%">
              <tr>
                <th style="text-align:center">Collection</th>
                <th style="text-align:center">Votes</th> 
                <th style="text-align:center">Comment</th>
              </tr>
              <tr>
                <td style="text-align:center">Bronx</td>
                <td style="text-align:center">11000</td> 
                <td style="text-align:center"></td>
              </tr>
              <tr>
                <td style="text-align:center">Queens</td>
                <td style="text-align:center">120000</td> 
                <td style="text-align:center"></td>
              </tr>
              <tr>
                <td style="text-align:center">Mail-In</td>
                <td style="text-align:center">56000</td> 
                <td style="text-align:center"></td>
              </tr>
            </table>

            <p>with one header line, as shown, then one data line for
            each collection of paper ballots that may be sampled.
            Each data line gives the name of the collection and
            then the number of cast votes in that collection
            (that is, the number of cast paper ballots in the collection).
            An additional column for optional comments is provided.</p>

            
            Collections File: <input type="file" name="collections" />
        
            <h2>Step 2: Upload Reported Votes File</h2>

            <p> In the box below, upload a CSV file with the data about the reported votes
            in the contest being audited. </p>

            <h3>
            REPORTED.CSV format:</h3>

            <p>Example reported.csv file:</p>

            <table style="width:100%">
              <tr>
                <th style="text-align:center">Collection</th>
                <th style="text-align:center">Reported</th> 
                <th style="text-align:center">Votes</th> 
                <th style="text-align:center">Comment</th>
              </tr>
              <tr>
                <td style="text-align:center">Bronx</td>
                <td style="text-align:center">Yes</td> 
                <td style="text-align:center">8500</td> 
                <td style="text-align:center"></td>
              </tr>
              <tr>
                <td style="text-align:center">Bronx</td>
                <td style="text-align:center">No</td> 
                <td style="text-align:center">2500</td> 
                <td style="text-align:center"></td>
              </tr>
              <tr>
                <td style="text-align:center">Queens</td>
                <td style="text-align:center">Yes</td> 
                <td style="text-align:center">100000</td> 
                <td style="text-align:center"></td>
              </tr>
              <tr>
                <td style="text-align:center">Queens</td>
                <td style="text-align:center">No</td> 
                <td style="text-align:center">20000</td> 
                <td style="text-align:center"></td>
              </tr>
              <tr>
                <td style="text-align:center">Mail-In</td>
                <td style="text-align:center">Yes</td> 
                <td style="text-align:center">5990</td> 
                <td style="text-align:center"></td>
              </tr>
              <tr>
                <td style="text-align:center">Mail-In</td>
                <td style="text-align:center">No</td> 
                <td style="text-align:center">50000</td> 
                <td style="text-align:center"></td>
              </tr>
              <tr>
                <td style="text-align:center">Mail-In</td>
                <td style="text-align:center">Overvote</td> 
                <td style="text-align:center">10</td> 
                <td style="text-align:center"></td>
              </tr>
            </table>

            <p>with one header line, as shown, then one data line for each
            collection and each reported choice in that collection,
            giving the number of times such reported choice is reported
            to have appeared.  An additional column for optional comments is
            provided.</p>

            <p>A reported choice need not be listed, if it occurred zero times,
            although every possible choice (except write-ins) should be listed
            at least once for some contest, so that this program knows what the
            possible votes are for this contest.</p>

            <p>For each collection, the sum, over reported choices,  of the Votes given
            should equal the Votes value given in the corresponding line in
            the collections.csv file.</p>

            <p>Write-ins may be specified as desired, perhaps with a string
            such as "Write-in:Alice Jones".</p>

            <p>For ballot-polling audits, use a reported choice of "-MISSING"
            or "-noCVR" or any identifier starting with a "-".  (Tagging
            the identifier with an initial "-" prevents it from becoming
            elegible for winning the contest.)</p>
            
            Reported Votes File: <input type="file" name="reported_votes" />

            <h2>Step 3: Upload Sampled Votes File</h2>

            <p> In the box below, upload a CSV file with the data about the samples
            from the contest being audited. </p>

            <h3>
            SAMPLE.CSV format:</h3>

            <p>Example sample.csv file:</p>

            <table style="width:100%">
              <tr>
                <th style="text-align:center">Collection</th>
                <th style="text-align:center">Reported</th> 
                <th style="text-align:center">Actual</th> 
                <th style="text-align:center">Votes</th> 
                <th style="text-align:center">Comment</th>
              </tr>
              <tr>
                <td style="text-align:center">Bronx</td>
                <td style="text-align:center">Yes</td> 
                <td style="text-align:center">Yes</td> 
                <td style="text-align:center">62</td> 
                <td style="text-align:center"></td>
              </tr>
              <tr>
                <td style="text-align:center">Bronx</td>
                <td style="text-align:center">No</td> 
                <td style="text-align:center">No</td> 
                <td style="text-align:center">21</td> 
                <td style="text-align:center"></td>
              </tr>
              <tr>
                <td style="text-align:center">Bronx</td>
                <td style="text-align:center">No</td> 
                <td style="text-align:center">Overvote</td> 
                <td style="text-align:center">1</td> 
                <td style="text-align:center"></td>
              </tr>
              <tr>
                <td style="text-align:center">Queens</td>
                <td style="text-align:center">Yes</td> 
                <td style="text-align:center">Yes</td> 
                <td style="text-align:center">504</td> 
                <td style="text-align:center"></td>
              </tr>
              <tr>
                <td style="text-align:center">Queens</td>
                <td style="text-align:center">No</td> 
                <td style="text-align:center">No</td> 
                <td style="text-align:center">99</td> 
                <td style="text-align:center"></td>
              </tr>
              <tr>
                <td style="text-align:center">Mail-In</td>
                <td style="text-align:center">Yes</td> 
                <td style="text-align:center">Yes</td> 
                <td style="text-align:center">17</td> 
                <td style="text-align:center"></td>
              </tr>
              <tr>
                <td style="text-align:center">Mail-In</td>
                <td style="text-align:center">No</td> 
                <td style="text-align:center">No</td> 
                <td style="text-align:center">73</td> 
                <td style="text-align:center"></td>
              </tr>
              <tr>
                <td style="text-align:center">Mail-In</td>
                <td style="text-align:center">No</td> 
                <td style="text-align:center">Lizard People</td> 
                <td style="text-align:center">2</td> 
                <td style="text-align:center"></td>
              </tr>
              <tr>
                <td style="text-align:center">Mail-In</td>
                <td style="text-align:center">No</td> 
                <td style="text-align:center">Lizard People</td> 
                <td style="text-align:center">1</td> 
                <td style="text-align:center"></td>
              </tr>
            </table>

            <p>with one header line, as shown then at least one data line for
            each collection for each reported/actual choice pair that was seen
            in the audit sample at least once, giving the number of times it
            was seen in the sample.  An additional column for comments is
            provided.</p>

            <p>There is no need to give a data line for a reported/actual pair
            that wasn't seen in the sample.</p>

            <p>If you give more than one data line for a given collection,
            reported choice, and actual choice combination, then the votes
            are summed. (So the mail-in collection has three ballots the scanner said
            showed "No", but that the auditors said actually showed "Lizard
            People".)  For example, you may give one line per audited ballot,
            if you wish.</p>

            <p>The lines of this file do not need to be in any particular order.
            You can just add more lines to the end of this file as the audit
            progresses, if you like.</p>

            Sample Votes File: <input type="file" name="sample" />

            <h2>(Optional) Specify random number seed</h2>
            <p>
            The computation uses a random number seed, which defaults to 1.
            You may if you wish enter a different seed here.
            (Using the same seed with the same data always returns the same results.)
            This is an optional parameter; there should be no reason to change it.
            </p>

            Seed: <input type="text" name="seed" />

            <h2>(Optional) Specify number of trials</h2>
            <p>Bayesian audits work by simulating the data which hasn't been sampled to
            estimate the chance that each candidate would win a full hand recount.
            You may specify in the box below the number of 
            trials used to compute these estimates.
            This is an optional parameter, defaulting to 10000.  Making it smaller
            will decrease accuracy and improve running time; making it larger will
            improve accuracy and increase running time. 
            </p>
         
            Number of trials:<input type="text" name="num_trials" />

            <h2>Compute results</h2>
            Click on the "Submit" button below to compute the desired answers,
            which will be shown on a separate page.
            <input type="submit" />

            Note: The Bayesian prior is represented by a pseudocount of one vote for
            each choice.  This may become an optional input parameter later.
            </form>
            '''

    @cherrypy.expose
    def ComparisonAudit(
        self, collections, reported_votes, sample, seed, num_trials):

        if seed == '':
            seed = 1
        seed = int(seed)

        if num_trials == '':
            num_trials = 25000
        num_trials = int(num_trials)

        if not os.path.exists("tmp_files"):
            os.mkdir("tmp_files")

        collections_filename = 'tmp_files/' + collections.filename
        reported_votes_filename = 'tmp_files/' + reported_votes.filename
        sample_filename = 'tmp_files/' + sample.filename
   
        collections_data = collections.file.read()
        collections_file = open(
            collections_filename, 'wb')
        collections_file.write(collections_data)
        collections_file.close()

        reported_data = reported_votes.file.read()
        reported_data_file = open(
            reported_votes_filename, 'wb')
        reported_data_file.write(reported_data)
        reported_data_file.close()

        sample_data = sample.file.read()
        sample_data_file = open(
            sample_filename, 'wb')
        sample_data_file.write(sample_data)
        sample_data_file.close()

        collection_names, collection_size, collections_rows = \
            bctool.read_and_process_collections(collections_filename)

        reported_choices, reported_size, reported_rows = \
            bctool.read_and_process_reported(reported_votes_filename,
                                             collection_names,
                                             collection_size,
                                             collections_filename)
        actual_choices, sample_dict, sample_rows = \
            bctool.read_and_process_sample(sample_filename,
                                           collection_names,
                                           reported_choices)

        shutil.rmtree("tmp_files")

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
                    # Default pseudocount_match = 50
                    stratum_pseudocounts.append(50)
                else:
                    # Default pseudocount base = 1
                    stratum_pseudocounts.append(1)
            strata_sample_tallies.append(np.array(stratum_sample_tally))
            strata_pseudocounts.append(np.array(stratum_pseudocounts))

        # Default value for n winners
        n_winners = 1

        win_probs = bctool.compute_win_probs(
                        strata_sample_tallies,
                        strata_pseudocounts,
                        strata_size,
                        seed,
                        num_trials,
                        actual_choices,
                        n_winners)
        return self.get_html_results(actual_choices, win_probs, n_winners)

    def get_html_results(self, actual_choices, win_probs, n_winners):
        """
        Given list of candidate_names and win_probs pairs, print summary
        of the Bayesian audit simulations.

        Input Parameters:

        -candidate_names is an ordered list of strings, containing the name of
        every candidate in the contest we are auditing.

        -win_probs is a list of pairs (i, p) where p is the fractional
        representation of the number of trials that candidate i has won
        out of the num_trials simulations.

        Returns:

        -String of HTML formatted results, which make a table on the online
        BPTool.
        """

        results_str = (
            '<style> \
            table, th, td { \
                     border: 1px solid black; \
            }\
            </style>\
            <h1> BCTOOL (Bayesian ballot-comparison tool version 0.8) </h1>')

        want_sorted_results = True
        if want_sorted_results:
            sorted_win_probs = sorted(
                win_probs, key=lambda tup: tup[1], reverse=True)
        else:
            sorted_win_probs = win_probs

        results_str += '<table style="width:100%">'
        results_str += '<tr>'
        results_str += ("<th>{:<24s}</th> <th>{:<s}</th>"
              .format("Choice",
                      "Estimated probability of winning a full manual recount"))
        results_str += '</tr>'

        for choice_index, prob in sorted_win_probs:
            choice = str(actual_choices[choice_index - 1])
            results_str += '<tr>'
            results_str += ('<td style="text-align:center">{:<24s}</td>').format(choice)
            results_str += ('<td style="text-align:center">{:6.2f} %</td>').format(100*prob)
            results_str += '</tr>'
        results_str += '</table>'
        results_str += '<p> Click <a href="./">here</a> to go back to the main page.</p>'
        return results_str



server_conf = os.path.join(os.path.dirname(__file__), 'server_conf.conf')

if __name__ == '__main__':

    # cherrypy.config.update({'tools.sessions.on': True,
    #                     'tools.sessions.storage_type': "File",
    #                     'tools.sessions.storage_path': 'sessions',
    #                     'tools.sessions.timeout': 10
    #            })
    cherrypy.quickstart(BCToolPage(), config=server_conf)
