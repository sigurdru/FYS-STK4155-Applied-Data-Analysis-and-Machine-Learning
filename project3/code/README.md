# Code of project 3
This README includes a description of code structure and every file in repo.

# Dataflow
As in the last two projects, the `main.py` is the only file run directly, and takes many commandline arguments. These arguments determine what functions are run, and parameter values. Run `main.py -h` to get a list of possible arguments and values. `main.py` calls on one of the three defined functions in `analysis.py`. One solves the diffusion equation with forward Euler, one with neural network and the last solves an eigenvalue problem with neural network. The results are plotted from `plot.py`. 

The makefile contains the commands used to generate all the plots in the rapport. Running `make all` should reproduce all results in the repport. This will take about 20min.

# File breakdown
## main.py
Startpoint of any run. Takes commandline arguments and configures run.

## analysis.py
Functions here are called from main.py, performing the runs and then calling plotting functions. 

## PINN.py
Contains our neural network class that solves the diffusion equation

## NN_eig.py
Contains our neural network class that solves eigenvalue problem

## plot.py
Has many plotting functions
