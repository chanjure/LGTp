#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
import multiprocessing as mp
import os, sys
from itertools import repeat

import LGTp as lgt

print("Configuration generator")
print("cpu count : ",os.cpu_count())
print("RNG type : ",np.random.default_rng()) 

N = 12
data_dir = './data/U1-%d/b080to120s40/' %(N)

# beta_list = np.linspace(0.95,1.05,20)
beta_list = np.linspace(0.8,1.2,40)


def simulate(b):
    seed = int(beta_list[b]*1000)

    u1 = lgt.Lattice([N,N,N,N])
    u1.init_fields('U1','Cold',seed)
		
    bare_parameters = u1.bare_parameter_generator()
		bare_parameters['beta'] = beta_list[b]

		g = lgt.action(u1,bare_parameters)

		# Calculate equilibrating time
    t_eq = lgt.calc_teq(bare_parameters,g.plaquetteSum_nb,u1)

		# Calculate exponential autocorrelation time
    t_ac = lgt.calc_tac(bare_parameters,g.plaquetteSum_nb,u1,t_eq=t_eq)
		t_ac = int(np.round(t_ac+0.5))
    
		# Exit if autocorrelation time exceeds 300
		if t_ac > 300:
			sys.exit("beta : %.3f t_ac : %.3f exiting" %(bare_parameters['beta'],t_ac))

		# Equilibrating phase
    for i in range(t_eq):
        lgt.metropolis(u1,bare_parameters)
    
		# Generate ensemble
    conf = []
        
    for e in range(n_conf*t_ac):
        lgt.metropolis(u1,bare_parameters)
        
        if not e%t_ac:
            conf.append(u1.field)
    
    beta = beta_list[b]
    conf_name = data_dir+'U1_b%0.3fN%dtac%d.npy' %(beta,N,t_ac)
    np.save(conf_name, conf)

# Generation settings
nt = len(beta_list)

ensem = []
n_conf = 200

print("Conf setting : N:%d")
print("data directory : "+data_dir)
print("number of conf per beta : %d"%(n_conf))

# test run
print("Starting test run")
start = time.time()
simulate(0)
dur = time.time() - start

n_ensem = len(beta_list)
n_core = os.cpu_count() # set cpu number
expected_dur = n_ensem*dur/n_core

print("test run duration : %.5f sec"%(dur))
print("for %d ensemble ~ %d sec ~ %0.3f hour"%(n_ensem,n_ensem*dur,n_ensem*dur/3600.))
print("with %d core, expecting : %0.3f hour"%(n_core, expected_dur/3600))


# Generate
print("Starting generation")
now = time.ctime(time.time())
expected_end = time.ctime(time.time() + expected_dur)

print("starting at "+now)
print("expected end time : "+expected_end)

start = time.time()

p = mp.Pool(n_core)
res = p.map(simulate, range(nt))
p.close()
p.join()

due = time.time() - start
print("time span:",due)

now = time.ctime(time.time())
print("ended at "+now)
