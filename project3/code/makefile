Euler_regression:
	python3 main.py -m Euler
	python3 main.py -m Euler -study_times -sc 0.5 -dx 0.01
	python3 main.py -m Euler -TE -sc 0.5 -dx 0.01
	python3 main.py -m Euler -TE -sc 0.5 -dx 0.1
NN_regression:
	python3 main.py -m NN -Nh 8 -Nn 20
	python3 main.py -m NN -Nh 4 -Nn 20
	python3 main.py -m NN -Nh 2 -Nn 20
	python3 main.py -m NN -Nh 4 -Nn 20 -Nt 500
	python3 main.py -m NN -Nh 4 -Nn 20 -Nt 1000
	python3 main.py -m NN -Nh 4 -Nn 20 -Nt 10000
	python3 main.py -m NN -Nh 4 -Nn 20 -Nt 50000
NN_eig:
	python3 main.py -m Eig -T 5 -N 1000
	python3 main.py -m Eig -T 5 -N 10
	python3 main.py -m Eig -T 5 -N 6
	python3 main.py -m Eig -T 5 -N 1000 -eta 0.1
	python3 main.py -m Eig -T 5 -N 1000 -eta 0.1 -epochs 50
	python3 main.py -m Eig -T 2 -N 1000 -dim 6

all: Euler_regression NN_regression NN_eig