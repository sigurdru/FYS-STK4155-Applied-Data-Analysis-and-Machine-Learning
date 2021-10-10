exercise1:
	python3 plot.py
	python3 main.py -p 5
exercise2:
	python3 main.py -r Bootstrap -ri 90 -p 20
	python3 main.py -r Bootstrap -ri 90 -p 10
	python3 main.py -r Bootstrap -ri 90 -p 15 -a bias_var_tradeoff
	python3 main.py -r Bootstrap -ri 90 -p 10 -a bias_var_tradeoff -e 0.5 
	python3 main.py -r Bootstrap -ri 90 -p 10 -a bias_var_tradeoff -e 0.1 
exercise3:
	python3 main.py -r CV -ri 5 -p 6
	python3 main.py -r CV -ri 7 -p 6 
	python3 main.py -r CV -ri 10 -p 6 
	python3 main.py -r CV -ri 10 -p 10
exercise4:
	python3 main.py -m Ridge -r Bootstrap -ri 90 -a lambda_analysis -l m6,2,51 -lc range -p 1,7
	python3 main.py -m Ridge -r Bootstrap -ri 90 -a lambda_analysis -l m6,2,51 -lc range -p 1,3,5,7,9 -pc list
	python3 main.py -m Ridge -r Bootstrap -ri 90 -a simple_regression -l 0.0001 -lc value -p 20
	python3 main.py -m Ridge -r Bootstrap -ri 90 -a simple_regression -l 0.01 -lc value -p 20
	python3 main.py -m Ridge -r Bootstrap -ri 90 -a simple_regression -l 1 -lc value -p 20
	python3 main.py -m Ridge -r Bootstrap -ri 90 -a simple_regression -l 100 -lc value -p 20
	# python3 main.py -a BVT_lambda -m Ridge -r Bootstrap -ri 90 -p 20 -l 1e-8,1e-4,1e2 -lc list  # This should be open in full screen before saving
	python3 main.py -m Ridge -r CV -ri 10 -a lambda_analysis -l m6,2,51 -lc range -p 1,7
	python3 main.py -m Ridge -r CV -ri 10 -a lambda_analysis -l m8,6,101 -lc range -p 1,3,5,7,9 -pc list
	python3 main.py -m Ridge -r CV -ri 10 -a simple_regression -l 0.0001 -lc value -p 10
	python3 main.py -m Ridge -r CV -ri 10 -a simple_regression -l 0.01 -lc value -p 10
	python3 main.py -m Ridge -r CV -ri 10 -a simple_regression -l 1 -lc value -p 10
	python3 main.py -m Ridge -r CV -ri 10 -a simple_regression -l 100 -lc value -p 10
exercise5:
exercise6:
	python3 main.py -d SRTM -r Bootstrap -ri 50 -n 30 -p 15 -log
	python3 main.py -d SRTM -r Bootstrap -ri 50 -n 50 -p 30 -log

all: oppgave1 oppgave2 oppgave3 oppgave4 oppgave5 oppgave6


