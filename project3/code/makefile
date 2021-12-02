Euler_regression:
	python3 main.py -m Euler
	python3 main.py -m Euler -TE -sc 0.5 -dx 0.01
	python3 main.py -m Euler -TE -sc 0.5 -dx 0.1
NN_regression:
	python3 main.py -m NN -Nh 8 -Nn 20
	python3 main.py -m NN -Nh 4 -Nn 20
	python3 main.py -m NN -Nh 2 -Nn 20

all: Euler_regression NN_regression