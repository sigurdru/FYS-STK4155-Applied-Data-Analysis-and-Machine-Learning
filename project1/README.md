# FYS-STK4155 Project1 - Regression analysis and resampling methods
The work of Håkon Olav Torvik, Vetle Viknes and Sigurd Sørlie Rustad in FYS-STK4155 - Applied Data Analysis and Machine Learning, fall 2021.

The report Project1.pdf is found in the tex directory.

Code dependencies:
- Python 3.8.10
    - numpy, 1.18.5
    - imageio, 2.9.0
    - sklearn, 0.24.2
    - matplotlib, 3.2.1
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
Can either be done for all (this takes a lot of time), or individual exercises. For example
```
#Run all exercises
make all       
#Run exercise 1 
make exercise1 
```