#-*-coding: utf-8 -*-

# --------------- #
#  Action Module  #
# --------------- #

import numpy as np
import numba as nb
import time
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

	def DS(self, field, *args):
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
		
		self.lat_shape = lat.lat_shape
		self.lat_dim = lat.lat_dim
		self.lat_size = lat.lat_size

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

#		g = np.ones(self.lat_shape)

#	a = np.random.randint(self.lat_shape[0])
#	b = np.random.randint(self.lat_shape[1])
#	c = np.random.randint(self.lat_shape[2])
#	d = np.random.randint(self.lat_shape[3])

		phi = np.random.uniform(-np.pi,np.pi)
		g = np.exp(1j*phi)
		
		return g
	
	def S(self, field):
		
		N = self.lat_shape
		s = 0.

		for n in np.ndindex(tuple(N)):
			for mu in range(self.lat_dim):
				for nu in range(mu):
					s += (1. - self.plaquette(field,n,mu,nu)).real

		return s

	def S_nb(self, field):
		
		sp = self.plaquetteSum_nb(field,ave=False).real
		
		s = self.lat_size*(self.lat_dim*(self.lat_dim-1.)/2.) - sp
		
		return s

	def DS(self, field, mu, *args):
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
		N = self.lat_shape

		x = np.random.randint(0,N[0])
		y = np.random.randint(0,N[1])
		z = np.random.randint(0,N[2])
		t = np.random.randint(0,N[3])
		
		n = tuple([x,y,z,t])

		A = self.staple(field,n,mu)
		phi = np.random.uniform(-np.pi,np.pi)
		g = np.exp(1j*phi)
		
		dS  = np.sum((field[x,y,z,t,mu]*(g - 1.)*A).real)

		new_field[x,y,z,t,mu] *= g

		return dS, new_field
		
	def plaquette(self,field,n,mu,nu):

		_e = np.identity(self.lat_dim, dtype=int)
		
		N = self.lat_shape
		n = np.array(n)
		
		p = (field[tuple(n)+(mu,)]
				*field[tuple((n+_e[mu])%N)+(nu,)]
				*(field[tuple((n+_e[nu])%N)+(mu,)].conj())
				*(field[tuple(n)+(nu,)].conj())
				)
		
		return p

	def staple(self,field,n,mu):
		
		_e = np.identity(self.lat_dim, dtype=int)
		
		N = self.lat_shape
		A = 0.
		
		n = np.array(n)
		
		for nu in range(self.lat_dim):
			if nu != mu:
				A += (field[tuple((n+_e[mu])%N)+(nu,)]
						 *(field[tuple((n+_e[nu])%N)+(mu,)].conj())
						 *(field[tuple(n%N)+(nu,)].conj())
						 +(field[tuple((n+_e[mu]-_e[nu])%N)+(nu,)].conj())
						 *(field[tuple((n-_e[nu])%N)+(mu,)].conj())
						 *field[tuple((n-_e[nu])%N)+(nu,)]
						 )

		return A

	def plaquetteSum(self,field,ave=True):
		"""Normalized plaquette sum
		"""

		N = self.lat_shape
		sp = 0.
		
		for n in np.ndindex(tuple(N)):
			for nu in range(self.lat_dim):
				for mu in range(nu):
					sp += self.plaquette(field,n,mu,nu).real

		if ave:
			return sp/self.lat_size/(self.lat_dim*(self.lat_dim-1.)/2.)
		else:
			return sp

	def plaquetteSum_nb(self,field,ave=True):
		return self._plaquetteSum_nb(field,self.lat_dim,self.lat_size,ave)
	
	@staticmethod
	@nb.jit
	def _plaquetteSum_nb(field,lat_dim,lat_size,ave=True):

		#e = np.identity(lat_dim,dtype=int)
		e = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
		
		N = np.shape(field)[:-1]
		sp = 0.

		for n in np.ndindex(N):
			for nu in range(lat_dim):
				for mu in range(nu):
					dir_mu = ((n[0]+e[mu][0])%N[0]
							,(n[1]+e[mu][1])%N[1]
							,(n[2]+e[mu][2])%N[2]
							,(n[3]+e[mu][3])%N[3]
							)
					dir_nu = ((n[0]+e[nu][0])%N[0]
							,(n[1]+e[nu][1])%N[1]
							,(n[2]+e[nu][2])%N[2]
							,(n[3]+e[nu][3])%N[3]
							)
					
					plaq = (field[n+(mu,)]
							*field[dir_mu+(nu,)]
							*(np.conj(field[dir_nu+(mu,)]))
							*(np.conj(field[n+(nu,)]))
							)
					
					sp += plaq.real
					
		if ave:
			return sp/lat_size/(lat_dim*(lat_dim-1.)/2.)
		else:
			return sp

	def polyakovLoop(self,field):

		N = self.lat_shape
		p = 0. + 0.j

		for n in np.ndindex(tuple(N[:-1])):
			p_local = 1. + 0.j
			for t in range(N[-1]):
				p_local *= field[tuple(n)+(t,)+(self.lat_dim-1,)]
			p += p_local

		return p/self.lat_size*N[-1]
	
	def polyakovLoop_nb(self,field):
		return self._polyakovLoop_nb(field,self.lat_size,self.lat_dim)
	
	@staticmethod
	@nb.njit
	def _polyakovLoop_nb(field,lat_size,lat_dim):

		N = np.shape(field)[:-1]
		p = 0. + 0.j

		for n in np.ndindex(N[:-1]):
			p_local = 1. + 0.j
			for t in range(N[-1]):
				p_local *= field[n+(t,)+(lat_dim-1,)]
			p += p_local

		return p/lat_size*N[-1]

class U1_2d():

	def __init__(self, lat, bare_args):
		
		if not lat.bare_parameter_checker(bare_args):
			raise ValueError("Wrong bare parameter. Use bare_parameter_generator to use bare parameter template.")
		
		self.lat_shape = lat.lat_shape
		self.lat_dim = lat.lat_dim
		self.lat_size = lat.lat_size

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

#		g = np.ones(self.lat_shape)

#	a = np.random.randint(self.lat_shape[0])
#	b = np.random.randint(self.lat_shape[1])
#	c = np.random.randint(self.lat_shape[2])
#	d = np.random.randint(self.lat_shape[3])

		phi = np.random.uniform(-np.pi,np.pi)
		g = np.exp(1j*phi)
		
		return g
	
	def S(self, field):
		
		N = self.lat_shape
		s = 0.

		for n in np.ndindex(tuple(N)):
			for mu in range(self.lat_dim):
				for nu in range(mu):
					s += (1. - self.plaquette(field,n,mu,nu)).real

		return s

	def DS(self, field, mu, *args):
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
		N = self.lat_shape

		x = np.random.randint(0,N[0])
		t = np.random.randint(0,N[1])
		
		n = tuple([x,t])

		A = self.staple(field,n,mu)
		phi = np.random.uniform(-np.pi,np.pi)
		g = np.exp(1j*phi)
		
		dS  = np.sum((field[x,t,mu]*(g - 1.)*A).real)

		new_field[x,t,mu] *= g

		return dS, new_field
		
	def plaquette(self,field,n,mu,nu):

		_e = np.identity(self.lat_dim, dtype=int)
		
		N = self.lat_shape
		n = np.array(n)
		
		p = (field[tuple(n)+(mu,)]
				*field[tuple((n+_e[mu])%N)+(nu,)]
				*(field[tuple((n+_e[nu])%N)+(mu,)].conj())
				*(field[tuple(n)+(nu,)].conj())
				)
		
		return p

	def staple(self,field,n,mu):
		
		_e = np.identity(self.lat_dim, dtype=int)
		
		N = self.lat_shape
		A = 0.
		
		n = np.array(n)
		
		for nu in range(self.lat_dim):
			if nu != mu:
				A += (field[tuple((n+_e[mu])%N)+(nu,)]
						 *(field[tuple((n+_e[nu])%N)+(mu,)].conj())
						 *(field[tuple(n%N)+(nu,)].conj())
						 +(field[tuple((n+_e[mu]-_e[nu])%N)+(nu,)].conj())
						 *(field[tuple((n-_e[nu])%N)+(mu,)].conj())
						 *field[tuple((n-_e[nu])%N)+(nu,)]
						 )

		return A

	def plaquetteSum(self,field):
		"""Normalized plaquette sum
		"""

		N = self.lat_shape
		sp = 0.
		
		for n in np.ndindex(tuple(N)):
			for nu in range(self.lat_dim):
				for mu in range(nu):
					sp += self.plaquette(field,n,mu,nu).real

		return sp/self.lat_size/(self.lat_dim-1.)

	def plaquetteSum_nb(self,field):
		return self._plaquetteSum_nb(field,self.lat_dim,self.lat_size)
	
	@staticmethod
	@nb.jit
	def _plaquetteSum_nb(field,lat_dim,lat_size):

		#e = np.identity(lat_dim,dtype=int)
		e = np.array([[1,0],[0,1]])
		
		N = np.shape(field)[:-1]
		sp = 0.

		for n in np.ndindex(N):
			for nu in range(lat_dim):
				for mu in range(nu):
					dir_mu = ((n[0]+e[mu][0])%N[0]
							,(n[1]+e[mu][1])%N[1]
							)
					dir_nu = ((n[0]+e[nu][0])%N[0]
							,(n[1]+e[nu][1])%N[1]
							)
					
					plaq = (field[n+(mu,)]
							*field[dir_mu+(nu,)]
							*(np.conj(field[dir_nu+(mu,)]))
							*(np.conj(field[n+(nu,)]))
							)
					
					sp += plaq.real
					
		return sp/lat_size/(lat_dim-1.)

	def polyakovLoop(self,field):

		N = self.lat_shape
		p = 0. + 0.j

		for n in np.ndindex(tuple(N[:-1])):
			p_local = 1. + 0.j
			for t in range(N[-1]):
				p_local *= field[tuple(n)+(t,)+(self.lat_dim-1,)]
			p += p_local

		return p/self.lat_size*N[-1]

	def polyakovLoop_norm_nb(self,field):
		Pol = self._polyakovLoop_nb(field,self.lat_size,self.lat_dim)
		return np.sqrt((Pol*np.conj(Pol)).real)
	
	def polyakovLoop_nb(self,field):
		return self._polyakovLoop_nb(field,self.lat_size,self.lat_dim)
	
	@staticmethod
	@nb.njit
	def _polyakovLoop_nb(field,lat_size,lat_dim):

		N = np.shape(field)[:-1]
		p = 0. + 0.j

		for n in np.ndindex(N[:-1]):
			p_local = 1. + 0.j
			for t in range(N[-1]):
				p_local *= field[n+(t,)+(lat_dim-1,)]
			p += p_local

		return p/lat_size*N[-1]

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
	elif lat.field_type == "U1-2d":
		act = U1_2d(lat, bare_args)
	else:
		raise ValueError("Wrong action type.")

	return act
