#-*-coding: utf-8 -*-

# -------------------- #
#  Monte Carlo Module  #
# -------------------- #

import numpy as np
from . import action
#import err

# TODO Make it module class later
#class MC:
#	"""Monte Carlo Module
#
#	This is module takes lattice object as an input and performs single Monte Carlo step.
#
#	Example
#	-------
#
# Notes
# -----
#
# Attributes
# ----------
#
# """
#
#	def __init__(self, ):

#def autocorrelation():


def metropolis(latt, action_type, sweep=None, bare_args=None):
	"""Metropolis algorithm

	Note
	----

	Parameters
	----------
	latt Lattice
		lattice to update
	action_type str
		action type
	discard int
		number of discard step
	args : dic
		arguments for action

	Returns
	-------
	tuple (np.array, bool)
		np.array Lattice is a updated lattice
		bool accept is 1 if accepted 0 if not.
		
	"""

	# MC parameter settings
	G = action(latt, bare_args)
	beta = bare_args["beta"]
	#TODO add sweep scheme as an option
	if sweep == None:
		sweep = 1
		for i in range(latt.dim):
			sweep *= latt.lat_shape[i]

	old_field = latt.field
	new_field = None

	S_diff = 0.
	accept = 0

	for i in range(sweep):
		g = G.transform()
		new_field = old_field*g
		S_diff += G.DS(old_field, new_field)

	r = np.random.uniform(0,1)
	Ta = min(1, np.exp(-beta*S_diff))

	if Ta >= r:
		accept = 1
		old_field = new_field

	return (old_field, accept)
		
	

		
