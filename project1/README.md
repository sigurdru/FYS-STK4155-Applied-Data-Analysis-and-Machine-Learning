# FYS-STK4155 Project1 - Regression analysis and resampling methods
The work of Håkon Olav Torvik, Vetle Viknes and Sigurd Sørlie Rustad in FYS-STK4155 - Applied Data Analysis and Machine Learning, fall 2021.

The report Project1.pdf is found in the tex directory.

Code dependencies:
- Python (runs with python version 3.8.10 and newer)
    - numpy, 1.19.4
    - matplotlib, 3.3.1
    - pandas, 1.1.4

## How to run the code
```
# move into the code directory
cd code
```

The code is now run by calling the program `main.py`. The arguments are handled
by the `argparse` module and directions will be printed to the terminal by
running the program with the `-h` flag (or `--help`):
```
python3 main.py -h
```

Notably, the `-r` flag (or `--results`) is for producing the results in the
report (both calculation and plotting): !!!DETTE VIRKER IKKE ENDA!!!
```
python3 main.py -r
```

### Code tests
All code tests are implemented in the program `test.py`, run it with the command
`python3 test.py`

!!!HER SIER VI HVA VI TESTER!!!

### Example run
The following will reproduce all the results in the report:
```
cd code
python3 main.py -r

# to run a specific calulation, for example
python3 main.py -p 5 -pc range -n 50 -m OLS
```
