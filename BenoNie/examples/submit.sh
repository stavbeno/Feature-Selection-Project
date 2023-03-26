#!/bin/bash


source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate ProjectA

python /home/shalev.shaer/ProjectA/Project_A/BenoNie/examples/synthetic_experiment.py $seed
#python /home/shalev.shaer/python_projects/hrt_project/main_for_semi.py $c $seed $rho $lmbda
