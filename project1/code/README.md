# Code of project 1
Will include a description of code structure, every file and every function.

# Dataflow
Our code runs in a very down-and-up-again way, starting at the top with `main.py`, moving deeper until `regression.py` is reach, where it moves back out to plot the data. The code is run by calling `main.py` with various arguments. One of the options is `-a, --analyse`, where the argument is the name of a function defined in `analysis.py`. Each does a different kind of analysis and generates various plots. Each generates data, and then calls upon one of the 3 implemented resampling functions implemented in `resampling.py`. These in turn call upon one of the 3 different regression functions implemented in `regression.py`. The regression methods return the beta-parametes to the resampling methods, which in turn return a data-dictionary to the analysis function which then calls upon the relevant plotingfunctions in `plot.py`. At most steps, the argparse-object is passed along, or at least the relevant parts of it.

# TODO

## Exersice 1: OLS
### Code
- [x] Implement OLS
- [ ] find confidence intervals of parameters beta
- [x] find MSE
- [x] find R2
- [x] Implement splitting
- [x] Implement Scaling

### Rapport
- [ ] Method
- [ ] Results
- [ ] Discuss why we scale

## Exersice 2: Bootstrap
### Code
- [x] Implement Bootstrapping
- [x] BVT-trade-off
- [ ] Make BVT tradeoff work propperly

### Rapport
- [ ] Remake figure 2.11 in Hasite et. al.
- [ ] Show expectation value rewrite
- [ ] Explain what terms mean
- [ ] Discuss trade-off

## Exersice 3: Cross-validation
### Code
- [ ] Implement CV
- [ ] Compare with SKlearn

### Rapport
- [ ] Method
- [ ] results

## Exersice 4: Ridge
**To me continued...**
