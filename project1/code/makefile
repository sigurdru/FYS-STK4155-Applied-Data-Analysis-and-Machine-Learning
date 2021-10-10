all: figure23 figure4a figure4b figure5 figure6a figure6b figure7

figure23:
	python3 main.py --show

figure4a:
	python3 main.py -r Bootstrap -ri 90 -p 20 --show

figure4b:
	python3 main.py -r Bootstrap -ri 90 -p 10 --show

figure5:
	python3 main.py -r Bootstrap -ri 90 -p 10 --show -a bias_var_tradeoff

figure6a: 
	python3 main.py -r Bootstrap -ri 90 -p 10 --show -a bias_var_tradeoff -e 0.1 

figure6b: 
	python3 main.py -r Bootstrap -ri 90 -p 10 --show -a bias_var_tradeoff -e 0.5 

figure7a:
	python3 main.py -r CV -ri 5 -p 6 --show 

figure7b:
	python3 main.py -r CV -ri 7 -p 6 --show 

figure7c:
	python3 main.py -r CV -ri 10 -p 6 --show 

figure7d:
	python3 main.py -r CV -ri 10 -p 10 --show 

figure7: figure7a figure7b figure7c figure7d
