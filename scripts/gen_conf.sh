#!/bin/bash

# Next:
# N = 6
# Nt = 6
# beta_id = b080to120s30
# n_conf=500

N=6
N_t=6
run_n=6
beta_id="b080to120s30"
n_conf=500

work_dir=../
data_dir=${work_dir}/data/conf/U1/4d/run${run_n}/U1-${N}/${beta_id}/
log_dir=${data_dir}/logs/
plot_dir=${data_dir}/plots/

mkdir -p ${data_dir}
mkdir -p ${log_dir}
mkdir -p ${plot_dir}

nohup python3 ./U1_auto_conf_gen.py $N $N_t $run_n $beta_id $n_conf $data_dir $plot_dir > ${log_dir}/u1-${N}gen.log &
