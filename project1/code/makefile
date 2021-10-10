exercise1:
	python3 plot.py
	python3 main.py
exercise2:
	python3 main.py -r Bootstrap -ri 90 -p 20
	python3 main.py -r Bootstrap -ri 90 -p 10
	python3 main.py -r Bootstrap -ri 90 -p 10 -a bias_var_tradeoff
	python3 main.py -r Bootstrap -ri 90 -p 10 -a bias_var_tradeoff -e 0.5 
	python3 main.py -r Bootstrap -ri 90 -p 10 -a bias_var_tradeoff -e 0.1 
exercise3:
	python3 main.py -r CV -ri 5 -p 6
	python3 main.py -r CV -ri 7 -p 6 
	python3 main.py -r CV -ri 10 -p 6 
	python3 main.py -r CV -ri 10 -p 10
exercise4:
exercise5:
exercise6:
	python3 main.py -d SRTM -r Bootstrap -ri 50 -n 30 -p 15 -log
	python3 main.py -d SRTM -r Bootstrap -ri 50 -n 50 -p 30 -log
	python3 main.py -d SRTM -r CV -ri 5 -n 50 -p 7 -log
	python3 main.py -d SRTM -r CV -ri 10 -n 50 -p 7 -log
	python3 main.py -d SRTM -r Bootstrap -ri 50 -n 50 -p 25 -a bias_var_tradeoff -log
	python3 main.py -d SRTM -r Bootstrap -ri 50 -n 30 -p 15 -a bias_var_tradeoff -log
	
all: exercise1 exercise2 exercise3 exercise4 exercise5 exercise6


