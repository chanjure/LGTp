#!/bin/bash

N=4
N_t=12
run_n=4
beta_id="b050to200s40"
n_conf=200

work_dir=../
data_dir=${work_dir}/data/conf/U1/4d/run${run_n}/U1-${N}/${beta_id}/
log_dir=${data_dir}/logs/

mkdir -p ${data_dir}
mkdir -p ${log_dir}

nohup python3 ./U1_auto_conf_gen.py $N $N_t $run_n $beta_id $n_conf > ${log_dir}/u1-${N}gen.log &
