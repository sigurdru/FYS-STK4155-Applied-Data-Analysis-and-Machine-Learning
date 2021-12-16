# Code of project 3 - additional exercise
This README includes a description of code structure and every file for the bias-variance analysis.

# Dataflow
As in the last two projects, the `bias_variance_main.py` is the only file run directly, and takes many commandline arguments. These arguments determine what functions are run, and parameter values. Run `bias_variance_main.py -h` to get a list of possible arguments and values. `bias_variance_main.py` calls on one of the three defined functions in `bias_variance_analysis.py`. One runs linear regression, another for neural network and a third for support vector machine. The results are plotted from `bias_variance_plot.py`. 

The makefile contains the commands used to generate all the plots in the exercise. Running `make all` should reproduce all results in the repport. This will take between 5 and 10 minutes.

# File breakdown
## bias_variance_main.py
Startpoint of any run. Takes commandline arguments and configures run.

## bias_variance_analysis.py
Functions here are called from bias_variance_main.py, performing the runs and then calling plotting functions. 

## bootstrap.py
Contains the Bootstrap class that runs a bootstrap simulation of desired method.

## bias_variance_utils.py
Utilities used in bootstrap simulation of the machine learning methods.

## bias_variance_plot.py
Plots the calculations from the analysis.
