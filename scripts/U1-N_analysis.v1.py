#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import cupy as cp
import time

import glob

import LGTp as lgt

N = 6

print("Starting Analysis U1-%d" %(N))

work_dir = "./"
data_dir = "./data/conf/U1/4d/U1-%d/b080to120s40/"%(N)
plot_dir = None

u1 = lgt.Lattice([N,N,N,N])
u1.init_fields('U1','Cold')
bare_args = u1.bare_parameter_generator()
g = lgt.action(u1,bare_args)

# Choose beta range.
# beta_list = np.linspace(0.95,1.05,20)
beta_list = np.linspace(0.80,1.20,40)

print("Loading data from "+data_dir)

# Load ensembles from "data_dir"
conf_list = glob.glob(data_dir+'/*.npy')
config = []

for conf_id in conf_list:
    conf = np.load(conf_id)
    config.append(conf)

print("Loaded ensemble shape : ",np.shape(config))


def measure(b):
    Plq1 = Pol_r1 = Plq2 = 0.
    Pol_c1 = 0.+0.j
    
    beta = beta_list[b]
    for i in range(len(config[b])):
#     for i in range(2):

        n1 = len(config[b])
        n2 = n1*n1
        
        Plaq = g.plaquetteSum_nb(config[b][i])
        Polya_c = g.polyakovLoop_nb(config[b][i])
        Polya = np.sqrt(Polya_c*np.conj(Polya_c))
#         print(Plaq, Polya_c, Polya)

        Plq1 += Plaq
        Pol_r1 += Polya
        Pol_c1 += Polya_c
        Plq2 += Plaq*Plaq
        
    Plq = Plq1/n1
    C = beta*beta*(Plq2/n1 - Plq1*Plq1/n2)
    Pol_r = Pol_r1/n1
    Pol_c = Pol_c1/n1
    
    return b, Plq, C, Pol_r, Pol_c

# test run
print("Starting test run")
start = time.time()
measure(0)
dur = time.time() - start

n_ensem = len(config)
n_core = 2
expected_dur = n_ensem*dur/n_core

print("test run duration : %.5f sec"%(dur))
print("for %d ensemble ~ %d sec ~ %0.3f hour"%(n_ensem,n_ensem*dur,n_ensem*dur/3600.))
print("with %d core, expecting : %0.3f hour"%(n_core, expected_dur/3600.))

# Measure
now = time.ctime(time.time())
expected_end = time.ctime(time.time() + expected_dur)

print("Starting measure")
print("starting at "+now)
print("expected end time : "+expected_end)

p = mp.Pool(n_core)
res = p.map(measure, range(len(beta_list)))
p.close()
p.join()
Plq, C, Pol_r = np.zeros(len(beta_list)), np.zeros(len(beta_list)), np.zeros(len(beta_list))
Pol_c = np.zeros(len(beta_list),dtype=np.complex128)
beta = np.zeros(len(beta_list))
for i, _r in enumerate(res):
    beta[i], Plq[i], C[i], Pol_r[i], Pol_c[i]= _r
    
true_end = time.ctime(time.time())
print("ended at "+true_end)

# Print result
print("========= Result =========")
print("beta	plq	C	pol_r	pol_c")
for i in range(len(beta)):
	print(beta[i],Plq[i],C[i],Pol_r[i],Pol_c[i]) 

if plot_dir is not None:

	f = plt.figure(figsize=(18, 10));

	sp =  f.add_subplot(2, 2, 1 );
	plt.scatter(beta_list, Plq, s=50, marker='o', color='IndianRed')
	plt.xlabel("Inverse coupling square beta", fontsize=20);
	plt.ylabel("Plaqquet sum ", fontsize=20);         plt.axis('tight');
	plt.grid(True)


	sp =  f.add_subplot(2, 2, 2 );
	plt.scatter(beta_list, Pol_r, s=50, marker='o', color='RoyalBlue')
	# plt.errorbar(beta_list, Pol_r, yerr=M_err, fmt='C0o')
	plt.xlabel("Inverse coupling square beta", fontsize=20); 
	plt.ylabel("|Polyacov loop| ", fontsize=20);   plt.axis('tight');
	plt.grid(True)


	sp =  f.add_subplot(2, 2, 3 );
	plt.scatter(beta_list, C, s=50, marker='o', color='IndianRed')
	plt.xlabel("Inverse coupling square beta", fontsize=20);  
	plt.ylabel("Specific Heat ", fontsize=20);   plt.axis('tight');   
	plt.grid(True)


	sp =  f.add_subplot(2, 2, 4 );
	plt.scatter(Pol_c.real, Pol_c.imag, s=50, marker='o', color='RoyalBlue')
	plt.xlabel("Re(P)", fontsize=20); 
	plt.ylabel("Im(P)", fontsize=20);   plt.axis('tight');
	plt.grid(True)

	plt.savefig(plot_dir+"U1-%d_analysis"+now+".png")
