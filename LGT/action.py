#-*-coding: utf-8 -*-

# --------------- #
#  Action Module  #
# --------------- #

import numpy as np
#import err

class Ising2d():
	
	def __init__(self, lat, bare_args):
		
		if not lat.bare_parameter_checker(bare_args):
			raise ValueError("Wrong bare parameter. Use bare_parameter_generator to use bare parameter template.")
		
		self.lat_shape = lat.lat_shape
		self.J = bare_args['J']
		self.h = bare_args['h']
		self.mu = bare_args['mu']
	
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
		#TODO : make it depend on self.dim

		g = np.ones(self.lat_shape)

		a = np.random.randint(self.lat_shape[0])
		b = np.random.randint(self.lat_shape[1])
		g[a,b] = -1
		
		return g
	
	def S(self, field):
		
		S = 0.
		N = self.lat_shape[0]
		M = self.lat_shape[1]

		for i in range(N):
			for j in range(M):
				s = field[i,j]
				nn = field[(i-1+N)%N][(j+M)%M] \
				+ field[(i+1+N)%N][(j+M)%M] \
				+ field[(i+N)%N][(j-1+M)%M] \
				+ field[(i+N)%N][(j+1+M)%M]

				S += -1.*(self.J*nn + self.mu*self.h)*s

		return S*0.5

	def DS(self, field):
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

		new_field = field.copy()
		N = len(new_field)

		a = np.random.randint(0,N)
		b = np.random.randint(0,N)
		s = field[a,b]
		nn = field[(a+1)%N,b] \
				+field[(a-1)%N,b] \
				+field[a,(b+1)%N] \
				+field[a,(b-1)%N]
		
		dS = 2.*(self.J*nn + self.h*self.mu)*s

		new_field[a,b] *= -1

		return dS, new_field

class U1():

	def __init__(self, lat, bare_args):
		
		if not lat.bare_parameter_checker(bare_args):
			raise ValueError("Wrong bare parameter. Use bare_parameter_generator to use bare parameter template.")
		
		pass

	def plaq():
		pass

def action(lat, bare_args):
	"""action class

	This is a action class.
	It contains functions using Ising model actions.
	Group transform and calculating actions are included here.

	Example
	-------

	Notes
	-----

	Attributes
	----------

	"""
	#TODO add needed bareparameter list printing
	if lat.field_type == "Ising2d":
		act = Ising2d(lat, bare_args)
	elif lat.field_type == "U1":
		act = U1(lat, bare_args)
	else:
		raise ValueError("Wrong action type.")

	return act
