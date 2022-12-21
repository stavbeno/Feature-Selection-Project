#!/bin/bash


for c in 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18
do
# manual parameters for a single run (without a for loop):
#c=1
#seed=0

# admm parmeters:
lmbda=10
rho=1

for seed in {1..100}
do
sbatch -c 2 --gres=gpu:0 -o slurm-test.out -J exp --export=c=$c,seed=$seed,rho=$rho,lmbda=$lmbda ./submit.sh
done
done
