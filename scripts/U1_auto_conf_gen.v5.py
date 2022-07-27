#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
import multiprocessing as mp
import os, sys
from itertools import repeat

import LGTp as lgt

print("default_rng:",np.random.default_rng())

cpu_count = os.cpu_count()
print("os cpu_count:",cpu_count)


# calculate equilibrating phase
# Change only here
#N = 6
#N_t = 12
#run_n = 4
#beta_id = "b050to200s40"
#n_conf = 200

if __name__ == '__main__':

	args = sys.argv # N, N_t, run_n, beta_id, n_conf, data_dir

	print('Argument : {}'.format(args))
	
	if len(args) != 8:
		raise SyntaxError("Check args : N, N_t, run_n, beta_id, prec, data_dir, fig_dir")

	N = int(args[1]) # Spatial lattice point number 
	N_t = int(args[2]) # Temporal lattice point number
	run_n = int(args[3]) # run id
	beta_id = str(args[4]) # beta set id
	prec = float(args[5]) # target precision
	data_dir = str(args[6]) # data save directory
	fig_dir = str(args[7]) # figure save directory

	start_b = float(beta_id[1:4])*0.01
	end_b = float(beta_id[6:9])*0.01
	steps = int(beta_id[-2:])

	max_steps = 500
	
	beta_list = np.linspace(start_b,end_b,steps)
	print("generating U1-%d "%(N)+beta_id)

	nt = len(beta_list)

	ensem = []

	# for b in range(nt):
	def simulate(b):
	#     start = time.time()
			#seed = int(beta_list[b]*1000)
			seed = int((time.time() % 1)*1000)

			u1 = lgt.Lattice([N,N,N,N_t])
			u1.init_fields('U1','Cold',seed)
			
			bare_parameters = u1.bare_parameter_generator()
			bare_parameters['beta'] = beta_list[b]
			
			g = lgt.action(u1,bare_parameters)
			O = g.polyakovLoopR_nb # Target observable

			t_eq, t_ac, _, _ = lgt.calc_teq_tac(bare_parameters,
					O, 
					u1, 
					tol=prec, 
					max_steps=max_steps, 
					verbose=True, 
					fig_dir=fig_dir, 
					use_lat=True)
			
			t_eq = int(np.round(t_eq+0.5))
			t_ac = int(np.round(t_ac+0.5))
			
			print("beta",beta_list[b]," teq : ",t_eq," tac : ",t_ac)
			
			if t_ac > max_steps:
					return

			# Finish thermalizing if t_eq > max_steps
			if t_eq > max_steps*3:
				rem_eq = max_steps*2
			else:
				rem_eq = t_eq - max_steps

			for i in range(rem_eq):
				lgt.metropolis(u1,bare_parameters)
			
			conf = []
					
			# Generate minimum number of configurations
			O_mean = O(u1.field)
			O_hist = []
			O_diff_hist = []
			for i in range(100):
				O_mean_old = O_mean
				
				for t in range(2*t_ac):
				#for t in range(t_ac):
					lgt.metropolis(u1,bare_parameters)
				conf.append(u1.field)

				O_hist.append(O(u1.field))
				O_mean = np.mean(O_hist)
				O_diff = np.abs(O_mean - O_mean_old)
				O_diff_hist.append(O_diff)

			# Generate conf of target precision
			while np.mean(O_diff_hist[-100:]) > prec and len(O_diff_hist) < max_steps*3:
				
				O_mean_old = O_mean
				
				for t in range(2*t_ac):
				#for t in range(t_ac):
					lgt.metropolis(u1,bare_parameters)
				conf.append(u1.field)

				O_hist.append(O(u1.field))
				O_mean = np.mean(O_hist)
				O_diff = np.abs(O_mean - O_mean_old)
				O_diff_hist.append(O_diff)
			
			beta = beta_list[b]
			conf_name = data_dir+'/U1_b%0.3fN%dtac%dS%d.npy' %(beta,N,t_ac,seed)
			np.save(conf_name, conf)

	# Test run
	print("starting test run")
	start = time.time()
	simulate(0)
	dur = time.time() - start

	n_ensem = len(beta_list)
	n_core = cpu_count
	expected_dur = n_ensem*dur/n_core

	print("test run duration : %.5f sec"%(dur))
	print("for %d ensemble ~ %d sec ~ %0.3f hour"%(n_ensem,n_ensem*dur,n_ensem*dur/3600.))
	print("with %d core, expecting : %0.3f hour"%(n_core, expected_dur/3600))


	now = time.ctime(time.time())
	expected_end = time.ctime(time.time() + expected_dur)

	print("starting at "+now)
	print("expected end time : "+expected_end)

	start = time.time()

	p = mp.Pool(n_core)
	res = p.map(simulate, range(nt)[1:])
	p.close()
	p.join()

	due = time.time() - start
	print("time span:",due)

	print(due/3600)

