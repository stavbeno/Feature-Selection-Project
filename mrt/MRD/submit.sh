#!/bin/bash


source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate MRD

python3 ./main.py $c $seed $rho $lmbda
