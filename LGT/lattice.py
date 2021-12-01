#-*-coding: utf-8 -*-

# ------------------ #
#  Lattice practice  #
# ------------------ #

import numpy as np
#import err

class Lattice:
	"""Lattice class

	This is a lattice class. It is used as a basic frame for
	any lattice types.

	Example
	-------

	Notes
	-----

	Attributes
	----------

	"""

	def __init__(self, lat_shape):
		"""Lattice initialization.

		Note
		----
		This is basically a list of fields.
		Fields are defined as np.array.
		For example, U(1) field would be np.array of shape [4,]
		which corresponds to dirac indices.
		ERR_ID = 0

		Parameters
		----------
		lat_type : str
			Type of lattice
		lat_shape : list
			Shape of the lattice

		"""

		self.lat_shape = lat_shape

		# Initialization paramter validation list
		#available_lat_type_list = ['Ising', 'U(1)', 'SU(3)']
		
		# Initialization step error control
		#if self.lat_type not in available_lat_type_list:
		#	raise ValueError('\n ERR_ID 0 : Unavailable lattice type. Available list : ', available_lat_type_list)

		# Set lattice object
		self.dim = len(lat_shape)
		#self.latt = [None]*lat_shape[0]
		#for s in lat_shape[1:]:
		#	self.latt = [self.latt]*s
		#TODO add rng selection

	def init_fields(self, field_type, init_scheme):
		"""Set fields on the lattice

		Note
		----
		ERR_ID = 1

		Parameters
		----------
		field_type str
			Type of field to initialize
		init_scheme str
			initialization scheme

		Returns
		-------

		"""

		self.field_type = field_type
		self.init_scheme = init_scheme
		
		# Initialization parameter validation list
		available_field_type_list = ['Ising', 'U(1)', 'SU(3)']
		available_init_scheme_list = ['Cold', 'Hot', 'man']

		# Initialization step error control
		if self.field_type not in available_field_type_list:
			raise ValueError('\n ERR_ID 1 : Unavailable field type. Available list : ', available_field_type_list)

		if self.init_scheme not in available_init_scheme_list:
			raise ValueError('\n ERR_ID 1 : Unavailable field initialization scheme. Available list : ', available_init_scheme_list)

		# Set field value at each lattice points.
		if self.field_type == 'Ising':
			if self.init_scheme == 'Cold':
				self.field = np.ones(self.lat_shape)
			elif self.init_scheme == 'Hot':
				self.field = 2*np.random.randint(0, 2, self.lat_shape) - 1
			#else:
			#	self.err(err_id=1, err_msg='Ising')
		
		elif self.field_type == 'U(1)':
			if self.init_scheme == 'Cold':
				self.field = np.ones(self.lat_shape + [dim])
			elif self.init_scheme == 'Hot':
				self.field = np.random.uniform(0., 1., self.lat_shape)
				#TODO change random range as a input parameter
			#else:
			# self.err(err_id=1, err_msg='U(1)')

	#def generate():
	# generate configuration
	# make test run function to calculate auto correlation
	# analyze autocorrelation (testrun) give information on
	# autocorrelation.
	# Then generate configurations using autocorrelation information
	# gained from testrun
	#TODO add option to calculate 
