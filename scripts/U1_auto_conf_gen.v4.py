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
		raise SyntaxError("Check args : N, N_t, run_n, beta_id, n_conf")

	N = int(args[1])
	N_t = int(args[2])
	run_n = int(args[3])
	beta_id = str(args[4])
	n_conf = int(args[5])
	data_dir = str(args[6])
	fig_dir = str(args[7])

	start_b = float(beta_id[1:4])*0.01
	end_b = float(beta_id[6:9])*0.01
	steps = int(beta_id[-2:])
	
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

			t_eq = lgt.calc_teq(bare_parameters, g.plaquetteSum_nb, u1, verbose=True, fig_dir=fig_dir)

			t_ac = lgt.calc_tac(bare_parameters, g.plaquetteSum_nb, u1, t_eq=t_eq, verbose=True, fig_dir=fig_dir)
			t_ac = int(np.round(t_ac+0.5))
			
			print("beta",beta_list[b]," with tac : ",t_ac)
			
			if t_ac > 300:
					return
			
			for i in range(t_eq):
					lgt.metropolis(u1,bare_parameters)
			
			conf = []
					
			for e in range(n_conf*t_ac):
					lgt.metropolis(u1,bare_parameters)
					
					if not e%t_ac:
							conf.append(u1.field)
			
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


	# In[ ]:


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

