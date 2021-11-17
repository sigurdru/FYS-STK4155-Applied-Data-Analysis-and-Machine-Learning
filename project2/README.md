# FYS-STK4155 Project2 - Using Regression and Neural Networks to Fit Continuous Functions and Classify Data
The work of Håkon Olav Torvik, Vetle Viknes and Sigurd Sørlie Rustad in FYS-STK4155 - Applied Data Analysis and Machine Learning, fall 2021.

The report [project2.pdf](https://github.com/sigurdru/FYS-STK4155/blob/main/project2/tex/project2.pdf) is found in the tex directory.

Code dependencies:
- Python 3.8.10
    - numpy, 1.18.5
    - pandas, 1.1.1
    - sklearn, 0.24.2
    - matplotlib, 3.2.1
    - seaborn, 0.11.2
    - tqdm, 4.62.3
    - imageio, 2.9.0
    - argparse, 1.1
## How to run the code
```
# move into the code directory
cd code
```

The README inside the code directory gives an overwiev of the files and dataflow. 
The code is run by calling the program `main.py`. It takes several commandline arguments,
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
#Run results for neural network on Cancer data
make NN_classification
```

Runnig all is expected to take between 10 and 30minutes, depending on the computer. A total time of 15.5 minutes has been achieved.
