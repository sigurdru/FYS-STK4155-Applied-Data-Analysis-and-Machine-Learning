exercise1:
	python3 plot.py
	python3 main.py
exercise2:
	python3 main.py -r Bootstrap -ri 90 -p 20
	python3 main.py -r Bootstrap -ri 90 -p 10
	python3 main.py -r Bootstrap -ri 90 -p 15 -a bias_var_tradeoff
exercise3:
	python3 main.py -r CV -ri 5 -p 6
	python3 main.py -r CV -ri 7 -p 6
	python3 main.py -r CV -ri 10 -p 6
	python3 main.py -r CV -ri 10 -p 10
exercise4:
	python3 main.py -m Ridge -r Bootstrap -ri 90 -a lambda_analysis -l m6,0,101 -lc range -p 1,15
	python3 main.py -m Ridge -r CV -ri 10 -a lambda_analysis -l m6,2,51 -lc range -p 1,7
	python3 main.py -a BVT_lambda -m Ridge -r Bootstrap -ri 90 -p 20 -l 1e-8,1e-4,1e2 -lc list
exercise5:
	python3 main.py -m Lasso -r Bootstrap -ri 90 -a lambda_analysis -l m6,m1,31 -lc range -p 1,11
	python3 main.py -a BVT_lambda -m Lasso -r Bootstrap -ri 90 -p 20 -l 1e-5,1e-4,1e2 -lc list
	python3 main.py -m Lasso -r CV -ri 10 -a lambda_analysis -l m2,1,51 -lc range -p 1,16
	python3 main.py -m Lasso -r CV -ri 10 -a lambda_analysis -l m6,m1,51 -lc range -p 1,7
exercise6:
	#Create OLS results
	python3 main.py -d SRTM -r Bootstrap -ri 50 -n 50 -p 30 -log
	python3 main.py -d SRTM -r Bootstrap -ri 50 -n 30 -p 15 -log
	python3 main.py -d SRTM -r CV -ri 5 -n 50 -p 7 -log
	python3 main.py -d SRTM -r CV -ri 10 -n 50 -p 7 -log
	python3 main.py -d SRTM -r Bootstrap -ri 50 -n 50 -p 25 -a bias_var_tradeoff -log
	python3 main.py -d SRTM -r Bootstrap -ri 50 -n 30 -p 15 -a bias_var_tradeoff -log
	#Create Ridge results
	python3 main.py -d SRTM -n 50 -m Ridge -r Bootstrap -ri 90 -a lambda_analysis -l m19,m2,31 -lc range -p 8,20
	python3 main.py -d SRTM -n 50 -m Ridge -r CV -ri 10 -a lambda_analysis -l m9,1,31 -lc range -p 1,10
	#Create Lasso results
	python3 main.py -d SRTM -n 50 -m Lasso -r Bootstrap -ri 90 -a lambda_analysis -l m10,m6,31 -lc range -p 4,12
	python3 main.py -d SRTM -n 50 -m Lasso -r CV -ri 10 -a lambda_analysis -l m9,m1,31 -lc range -p 5,15

all: exercise1 exercise2 exercise3 exercise4 exercise5 exercise6


