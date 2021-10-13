# Code of project 1
Will include a description of code structure, every file and every function.

# Dataflow
Our code runs in a very down-and-up-again way, starting at the top with `main.py`, moving deeper until `regression.py` is reached, where it moves back out and up to plot the data. The code is run by calling `main.py` with various arguments. One of the options is `-a, --analyse`, where the argument is the name of a function defined in `analysis.py`. Each does a different kind of analysis and generates various plots. Each generates data, and then calls upon one of the 3 implemented resampling functions implemented in `resampling.py`. These in turn call upon one of the 3 different regression functions implemented in `regression.py`. The regression methods return the beta-parametes to the resampling methods, which in turn return a data-dictionary to the analysis function which then calls upon the relevant ploting functions in `plot.py`.

Running `make all` should reproduce all results in rapport.

# File breakdown
## main.py
Startpoint of any run. Takes commandline arguments and configures run.

## analysis.py
Functions here are called from main.py, performing the runs and then calling plotting functions. All used to be quite simple and similar, but a quick bodge was needed to fix a bug.

## resampling.py
Contains functions for the 3 resampling methods, all taking the same arguments. Also contains some helper-functions.

## regression.py
Contains the 3 regression functions, all taking the same arguments.

## plot.py
Contains several plotting functions, making different figures. Also several helper-functions.
The plotting functions are written seperately, so title and filename creation is not at all similar between them.

## utils.py
Contains various functions, mostly related to creation of X and z.