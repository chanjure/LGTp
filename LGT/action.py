#-*-coding: utf-8 -*-

# --------------- #
#  Action Module  #
# --------------- #

import numpy as np
#import err

class action:
	"""action class

	This is a action class.
	It contains functions using actions.
	Group transform and calculating actions are included here.

	Example
	-------

	Notes
	-----

	Attributes
	----------

	"""
	#TODO add needed bareparameter list printing

	def __init__(self, latt, bare_args):
		"""Initialize action related parameters

		Note
		----
		
		ERR_ID = act_0

		Parameters
		----------
		lat_shape : list
			Shape of the lattice
		field_type : str
			Field type selects action
		bare_arg : dic
			dictionary of bare parameters.

		"""

		self.lat_shape = latt.lat_shape
		self.field_type = latt.field_type
		self.bare_args = bare_args

	def transform(self,):
		"""Generates symmetry group element.

		Note
		----
		Generated group element would be multiplied elementwise to the fields.

		ERR_ID = act_1

		Parameters
		----------

		Returns
		-------
		np.array 
			Symmetry group element
		"""

		g = np.ones(self.lat_shape)

		if self.field_type == 'Ising':

			a = np.random.randint(self.lat_shape[0])
			b = np.random.randint(self.lat_shape[1])
			g[a,b] = -1
		
		elif self.field_type == 'U(1)':
			g = None
		#else:
		#	err.err(err_id=1, err_msg='blah')

		return g

	def S(self, field):
		"""Calculate action
		
		Note
		----
		Calculate action

		ERR_ID = act_2

		Parameters
		----------
		field : dic
			bare parameters

		Returns
		-------
		float (or complex)
		"""

		#TODO: Make boundary condition option
		S = 0.
		N = self.lat_shape[0]
		M = self.lat_shape[1]

		if self.field_type == 'Ising':
			for i in range(N):
				for j in range(M):
					
					nn = field[(i-1+N)%N][(j+M)%M] \
					+ field[(i+1+N)%N][(j+M)%M] \
					+ field[(i+N)%N][(j-1+M)%M] \
					+ field[(i+N)%N][(j+1+M)%M] \

					S += -1.*(self.bare_args["J"]*nn + self.bare_args["mu"]*self.bare_args["h"])*field[(i+N)%N][(j+M)%M]

		elif self.field_type == 'U(1)':
			S = 0

		return S

	def DS(self, old_field, new_field):
		"""Difference in action

		Note
		----
		Calculates difference in action between new configuration and old configuration.

		ERR_ID = act_3

		Parameters
		----------
		new_field : np.array
		old_field : np.array

		Returns
		-------
		float
		"""

		dS = 0.

		if self.field_type == 'Ising':
			dS = self.S(old_field) - self.S(new_field)
		elif self.field_type == 'U(1)':
			dS = 0.
		#else:
		# err

		return dS

