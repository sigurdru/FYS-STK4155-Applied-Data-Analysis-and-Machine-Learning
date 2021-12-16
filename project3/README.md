# FYS-STK4155 Project3 - Using Forward Euler and Neural Networks to Solve Differential Equations
The work of Vetle Nevland, Vetle Viknes and Sigurd Sørlie Rustad in FYS-STK4155 - Applied Data Analysis and Machine Learning, fall 2021.

The report [project3.pdf](https://github.com/sigurdru/FYS-STK4155/blob/main/project3/tex/project3.pdf) is found in the tex directory.

The solution to the additional exercise is located in the folder [bias_variance](https://github.com/sigurdru/FYS-STK4155/tree/main/project3/bias_variance).

Code dependencies:
- Python 3.8.10
    - numpy, 1.18.5
    - matplotlib, 3.2.1
    - tqdm, 4.62.3
    - argparse, 1.1
    - tensorflow, 2.7.0
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

To reproduce all the results in our rapport, a makefile is configured to produce every plot.
Can either be done for all, or individual parts. For example
```
#Run all exercises
make all       
#Run results for neural network on diffusion equation
make NN_regression
```

Runnig all is expected to take between 10 and 30minutes, depending on the computer.
