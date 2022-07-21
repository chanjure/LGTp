#-*-coding: utf-8 -*-

# -------------------- #
#  Monte Carlo Module  #
# -------------------- #

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from . import Lattice
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
	lat Lattice
		lattice to update
	g Action
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
	g = action(lat,bare_args)
	beta = bare_args["beta"]
	
	#TODO add sweep scheme as an option
	accept = 0
	
	for mu in range(lat.aux_index):
		for i in range(lat.lat_size):
			dS, new_field = g.DS(lat.field,mu)

			if np.random.rand() < np.exp(-beta*dS):
				accept = 1
				lat.field = new_field

	return (lat.field, accept)

def calc_teq(bare_arg, O, init_lat, mcstep=metropolis, stride=10, tol=1e-2, max_step=300, verbose=0, fig_dir=None):

	t_eq = max_step

	O_cold = []
	O_hot = []
	
	lat_shape = init_lat.lat_shape
	field_type = init_lat.field_type
	seed = init_lat.seed

	cold_init = Lattice(lat_shape)
	cold_init.init_fields(field_type,'Cold',seed)

	hot_init = Lattice(init_lat.lat_shape)
	hot_init.init_fields(field_type,'Hot',seed)

	diff = 10*tol
	self_diff = 10*tol
	#cold_res_temp = 0.

	for i in range(max_step):
		mcstep(cold_init, bare_arg)
		mcstep(hot_init, bare_arg)

		cold_res = O(cold_init.field)
		hot_res = O(hot_init.field)

		#diff = np.abs(np.mean(cold_rs - hot_res)
		O_cold.append(cold_res)
		O_hot.append(hot_res)
		if i > 2*stride:
			diff = np.abs(np.average(O_cold[i-stride:i]) - np.average(O_hot[i-stride:i]))
			self_diff = np.abs(np.average(O_cold[i-2*stride:i-stride] - np.average(O_cold[i-stride:i])))

		#if diff < tol and np.abs(cold_res - cold_res_temp) < tol:
		if diff < tol and self_diff < tol:
			t_eq = i
			break

		#cold_res_temp = cold_res

	beta = bare_arg['beta']
	if verbose :
		plt.clf()
		plt.title(r"Estimation of thermalization time $1/e^2$=%.3f $\tau_{eq}$=%d"%(beta,t_eq), fontsize=15)
		plt.plot(np.arange(len(O_cold)),np.abs(O_cold),'C0.',label='cold start')
		plt.plot(np.arange(len(O_hot)),np.abs(O_hot),'C3.',label='hot start')
		plt.legend(loc='lower right',fontsize=12)
		plt.xlabel("Monte Carlo time",fontsize=12)
		plt.ylabel("Observable",fontsize=12)
		plt.grid(True)
		#plt.show()
		if fig_dir is not None:
			fig_title = fig_dir+"/b%.3fteq%.3f.png"%(bare_arg['beta'],t_eq)
			plt.savefig(fig_title,dpi=600)

	return t_eq

def autocorrelation(conf, O, t):
	N = len(conf) - t
	eps = 1e-7

	o1o2 = 0.
	o1 = 0.
	o2 = 0.

	o1o1 = 0.
	o2o2 = 0.

	for i in range(N):

		o1o2 += O(conf[i])*O(conf[i+t])/N

		o1 += O(conf[i])/N
		o2 += O(conf[i+t])/N

		o1o1 += O(conf[i])*O(conf[i])/N

	cor_t = o1o2 - o1*o2

	return cor_t/(o1o1 - o1*o1 + eps)

def calc_tac(bare_arg, O, init_lat, mcstep=metropolis, t_eq=100, n_conf_ac=500, verbose=False, fig_dir=None):

	def fit_func(x,a,b):
		return a*np.exp(-x/b)

	lat_shape = init_lat.lat_shape
	field_type = init_lat.field_type
	seed = init_lat.seed

	conf_ac = []

	ac_init = Lattice(lat_shape)
	ac_init.init_fields(field_type,'Cold',seed)

	for i in range(t_eq):
		mcstep(ac_init, bare_arg)

	for e in range(n_conf_ac):
		mcstep(ac_init, bare_arg)
		conf_ac.append(ac_init.field)

	ac_hist = np.empty(n_conf_ac)

	for i in range(n_conf_ac):
		ac_hist[i] = autocorrelation(conf_ac, O, i)
	
	fit_lim = int(n_conf_ac/5)
	fit_range = n_conf_ac - fit_lim
	fit_b, fit_cov = curve_fit(fit_func,np.arange(fit_range),ac_hist[:fit_range])
	
	tac_exp = fit_b[1]
	tac_int = 0.5 + np.sum(np.abs(ac_hist))

	beta = bare_arg['beta']
	if verbose :
		x = np.arange(n_conf_ac)
		y = fit_func(x,fit_b[0],fit_b[1])
		z = fit_func(x,fit_b[0],tac_int)

		plt.clf()
		plt.title(r"Autocorrelation plot $1/e^2$=%0.3f $\tau_{ac}$=%0.3f"%(beta,tac_int), fontsize=15)
		plt.plot(x[:fit_range], ac_hist[:fit_range], 'C0.', label="Autocorrelation")
		plt.plot(x[:fit_range], y[:fit_range],'C3.',label=r"Exponential fit $\tau_{exp}$=%0.3f"%(tac_exp))
		plt.plot(x[:fit_range], z[:fit_range],'C4.',label=r"Integrated fit $\tau_{int}$=%0.3f"%(tac_int))
		plt.xlabel("Monte Carlo time", fontsize=12)
		plt.ylabel("Autocorrelation", fontsize=12)
		plt.legend(loc="upper right",fontsize=12)
		plt.grid("True")
		#plt.show()
		if fig_dir is not None:
			fig_title = fig_dir+"/b%.3ftac%.3f.png"%(bare_arg['beta'],tac_int)
			plt.savefig(fig_title, dpi=600)

	return tac_exp

def _ac(O, t):
	N = len(O) - t
	eps = 1e-7

	o1o2 = 0.
	o1 = 0.
	o2 = 0.

	o1o1 = 0.
	o2o2 = 0.

	for i in range(N):

		o1o2 += O[i]*O[i+t]/N

		o1 += O[i]/N
		o2 += O[i+t]/N

		o1o1 += O[i]*O[i]/N

	cor_t = o1o2 - o1*o2

	return cor_t/(o1o1 - o1*o1 + eps)

def calc_teq_tac(bare_arg, O, init_lat, mcstep=metropolis, max_steps=500, tol=1e-5, verbose=False, fig_dir=None, use_lat=False):

	def tac_fit_func(x,a,b):
		return a*np.exp(-x/b)

	def teq_fit_func(x,a,b,c):
		return -a*np.exp(-x/b) + c

	lat_shape = init_lat.lat_shape
	field_type = init_lat.field_type
	seed = init_lat.seed

	if use_lat:
		lat = init_lat
	else:
		lat = Lattice(lat_shape)
		lat.init_fields(field_type,'Cold',seed)

	ac_hist = np.empty(max_steps)
	O_hist = np.empty(max_steps)

	for e in range(max_steps):
		mcstep(lat, bare_arg)
		O_hist[e] = O(lat.field)

	for i in range(max_steps):
		ac_hist[i] = _ac(O_hist, i)
	
	# teq fit
	# e^-x ~ tol => x ~ -ln(tol)
	teq_b, teq_cov = curve_fit(teq_fit_func,np.arange(max_steps),np.abs(O_hist))
	teq = teq_b[1]*(-np.log(tol/np.abs(teq_b[0])))

	# tac fit
	fit_lim = int(max_steps/5.)
	fit_range = max_steps - fit_lim
	tac_b, tac_cov = curve_fit(tac_fit_func,np.arange(fit_range),ac_hist[:fit_range])
	
	tac_exp = tac_b[1]
	tac_int = 0.5 + np.sum(np.abs(ac_hist))

	beta = bare_arg['beta']
	if verbose :
		plt.clf()
		x = np.arange(max_steps)

		f = plt.figure(figsize=(6.4,4.8*2))

		# Plot teq
		s_teq = f.add_subplot(2,1,1)
		s_teq.set_title(r"Estimation of thermalization time $1/e^2$=%.3f $\tau_{eq}$=%d"%(beta,teq), fontsize=15)

		s_teq.plot(x,np.abs(O_hist),'C0.',label='Observable')
		s_teq.plot(x,teq_fit_func(x,teq_b[0],teq_b[1],teq_b[2]),'C1.',label='Exponential fit')
		s_teq.axvline(teq,color='C3',linestyle='--',label=r'$\tau_{eq}=$%0.3f'%(teq))
		s_teq.legend(loc='lower right',fontsize=12)
		s_teq.set_xlabel("Monte Carlo time",fontsize=12)
		s_teq.set_ylabel("Observable",fontsize=12)
		s_teq.grid(True)

		# Plot tac
		y = tac_fit_func(x,tac_b[0],tac_b[1])
		z = tac_fit_func(x,tac_b[0],tac_int)

		s_tac = f.add_subplot(2,1,2)
		s_tac.set_title(r"Autocorrelation plot $1/e^2$=%0.3f $\tau_{ac}$=%0.3f"%(beta,tac_exp), fontsize=15)
		s_tac.plot(x[:fit_range], ac_hist[:fit_range], 'C0.', label="Autocorrelation")
		s_tac.plot(x[:fit_range], y[:fit_range],'C1.',label=r"Exponential fit $\tau_{exp}$=%0.3f"%(tac_exp))
		# s_tac.plot(x[:fit_range], z[:fit_range],'C2.',label=r"Integrated fit $\tau_{int}$=%0.3f"%(tac_int))
		s_tac.axvline(tac_exp,color='C3',linestyle='--', label=r'$\tau_{ac}=%.3f$'%(tac_exp))
		
		s_tac.set_xlabel("Monte Carlo time", fontsize=12)
		s_tac.set_ylabel("Autocorrelation", fontsize=12)
		s_tac.legend(loc="upper right",fontsize=12)
		s_tac.grid("True")

		f.tight_layout()
		
		if fig_dir is not None:
			fig_title = fig_dir+"/b%.3fteq%.3ftac%.3f.png"%(bare_arg['beta'],teq,tac_exp)
			plt.savefig(fig_title,dpi=600)

	return teq, tac_exp, O_hist, ac_hist
