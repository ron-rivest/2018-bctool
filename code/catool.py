def dirichlet_multinomial(sample_tally_with_cvrs, random_state, num_votes):
	"""
	Return a sample according to Dirichlet multinomial. We
	expect all the ballots in the tally to have matching CVRs, so we adjust
	the pseudocounts accordingly.
	"""
	pass

def generate_nonsample_tally(sample_tally_with_cvrs, random seed, num_votes):
	"""
	Use Dirichlet Multinomial to generate nonsample tally.
	"""
	pass

def compute_winner(sample_tally_with_cvrs, random seed, num_votes):
	"""
	Generate nonsample tally for a given sample tally and compute
	the winner of the election.
	"""
	pass

def compute_win_probs(sample_tally_with_cvrs, random seed, num_votes, num_trials):
	"""
	For the required number of trials, compute the winner of each
	trial and return the probability that each candidate is a winner.
	"""
	pass

def print_results(candidate_names, win_probs):
	"""
	Comparison-audit specific printing of what happened during the course
	of the audit.
	"""
	pass

def preprocess_csv(path_to_csv):
	"""
	TODO: What does a csv input file look like for a comparison audit?
	Look at audit-lab for what inputs are generated by syn2.py

	If they aren't too different, can we refactor this out into utils.py?
	"""
	pass

def main():
	"""
	Main loop to calculate the Bayesian risk for a comparison audit.
	Command-line arguments include:
	-total_num_votes (a number for a single county election)
	-single_county_tally_with_cvr (tuples of the form (votes, cvrs))?
	-path_to_csv (string, used when we want to preprocess from a csv)
	-audit_seed (number)
	-num_trials (number, required command line arg)
	-pseudocount_for_agreement (optional, defaults to 10)
	-pseudocount_for_disagreement (optional, defaults to 1)
	-pretty_print (optional, defaults to True)
	"""
	pass