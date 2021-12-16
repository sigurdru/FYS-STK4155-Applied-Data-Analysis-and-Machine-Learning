# FYS-STK4155 Project3 - Additional exercise

Code dependencies:
- Python 3.8.10
    - numpy, 1.18.5
    - matplotlib, 3.2.1
    - argparse, 1.1

## How to run the code
```
# To run the code of the main report move into the code directory
cd code
# To run the code for the additional exercise move into bias_variance/code_extra
cd bias_variance/code_extra
```

The README inside the code directories gives an overwiev of the files and dataflow.
All code is run by calling the program `main.py`. It takes several commandline arguments,
handled by the `argparse` module. Directions will be printed to the terminal by
running the program with the `-h` flag (or `--help`).  
```
python3 main.py -h
```

To reproduce all the results in of the exercise, a makefile is configured to produce every plot.
Can either be done for all, or individual parts. For example
```
#Run all exercises
make all       
#Run results for neural network on diffusion equation
make SVM
```

Runnig all is expected to take between 5 and 10 minutes, depending on the computer.
