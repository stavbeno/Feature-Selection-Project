# Maximum Risk Discrepancy (MRD)
This a repository for reproducing the main experiments from the paper "Learning to Increas the Power of Conditional Randomization Tests".
The code is adapted to work in SLURM environment.
To run it locally, you can run the main script and adapt the main function.
The main script include a function for each experiment, and you should choose what experiment you want to run by uncomment the desired experiment (and set the function's parameter as you want).
Each experiment produces a csv file including the p-values of each model and meta-data. All csvs can be analyzed in the analyze.ipynb script.

To fit MRD-Lasso on your own data, use lasso_admm. Note that the features must be sacled to have zero mean since it does not fit intercept. 


Attached a virtual environment file MRD.yml with all the packages that required to run this code. 
