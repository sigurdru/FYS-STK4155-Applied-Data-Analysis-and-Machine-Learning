# Code of project 2
This README includes a description of code structure and every file in repo.

# Dataflow
As in last project, the `main.py` is the only file run directly, and takes many commandline arguments. These arguments determine what functions are run, and parameter values. Run `main.py -h` to get a list of possible arguments and values. `main.py` calls on 1 of the 4 defined functions in `analyse.py`. One performs regression with neural network, one with SGD, one does classification with logistic regression and one with neural network. The results are plotted from `plot.py`. 

The makefile contains the commands used to generate all the plots in the rapport. Running `make all` should reproduce all results in rapport. This will take about 15min.

# File breakdown
## main.py
Startpoint of any run. Takes commandline arguments and configures run.

## analysis.py
Functions here are called from main.py, performing the runs and then calling plotting functions. 

## SGD.py
Contains code for regression and logistic regression with SGD

## NeuralNetwork.py
Contains out neural network class, as well as an optimizer

## cost_activation.py
Contains classes with the implemented cost and activation functions

## plot.py
Has many plotting functions

## utils.py
Contains various functions, mostly related to data loading.

## brest_cancer.py
File to study the breast cancer data and correlation matrix. Not used in rapport
