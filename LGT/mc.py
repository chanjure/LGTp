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


def metropolis(lat, bare_args={'beta':1}):
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
	G = action(lat, bare_args)
	beta = bare_args["beta"]
	
	#TODO add sweep scheme as an option
	sweep = 1
	for i in range(lat.dim):
		sweep *= lat.lat_shape[i]

	accept = 0

	for i in range(sweep):
		dS, new_field = G.DS(lat.field)

		if np.random.rand() < np.exp(-beta*dS):
			accept = 1
			lat.field = new_field

	return (lat.field, accept)
		
	

		
