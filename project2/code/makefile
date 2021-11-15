SGD_regression:
	python3 main.py -d Franke -m reg -Ne 150 -eta 'np.linspace(0.01,0.9,21)' 
	python3 main.py -d Franke -m reg -Ne 150 -eta 'np.linspace(0.01,0.5,21)'
	python3 main.py -d Franke -m reg -Ne 150 -eta 0.25 -bs 0 
	python3 main.py -d Franke -m reg -Ne 150 -eta 'np.linspace(0.1,0.5,11)' -l 'np.logspace(-5,0,11)'
	python3 main.py -d Franke -m reg -Ne 150 -eta 0.25 -ga -1
	python3 main.py -d Franke -m reg -Ne 150 -eta 'np.linspace(0.1,0.9,21)' -de 
NN_regression:
	python3 main.py -d Franke -p 1 -eta 'np.logspace(-4, 0, 11)' -l 'np.logspace(-5, -1, 11)' -Ne 100
	python3 main.py -d Franke -p 1 -eta 'np.logspace(-4, 0, 11)' -l 'np.logspace(-5, -1, 11)' -de -Ne 100
	python3 main.py -d Franke -p 1 -eta 'np.logspace(-4, 0, 11)' -l 'np.logspace(-5, -1, 11)' -de -Ne 100 -ga 0.5
	python3 main.py -d Franke -p 1 -eta 'np.logspace(-5, -1, 11)' -l 'np.logspace(-5, -1, 11)' -de -Ne 100 -act_func relu
	python3 main.py -d Franke -p 1 -eta 'np.logspace(-5, -1, 11)' -l 'np.logspace(-5, -1, 11)' -de -Ne 100 -act_func leaky_relu
NN_classification:
	python3 main.py -d Cancer -hn 20,20 -eta 'np.logspace(-4, 0, 11)' -l 'np.logspace(-5, 1, 11)' -bs 35 -act_func sigmoid
	python3 main.py -d Cancer -hn 20,20 -eta 'np.logspace(-4, 0, 11)' -l 'np.logspace(-5, 1, 11)' -bs 35 -de -act_func sigmoid
	python3 main.py -d Cancer -hn 20,20 -eta 'np.logspace(-4, 0, 11)' -l 'np.logspace(-5, 1, 11)' -bs 35 -de -act_func relu
	python3 main.py -d Cancer -hn 20,20 -eta 'np.logspace(-4, 0, 11)' -l 'np.logspace(-5, 1, 11)' -bs 35 -de -act_func leaky_relu
Logistic_regression:
	py main.py -d Cancer -m reg -bs 5 -eta 'np.logspace(-5, -0.5, 11)' -Ne 200
	py main.py -d Cancer -m reg -bs 5 -eta 'np.logspace(-5, -1, 11)' -l 'np.logspace(-5, -3, 11)' -Ne 200

all: SGD_regression NN_regression NN_classification Logistic_regression

test:
	python3 main.py -d Cancer -hn 20,20 -eta 'np.logspace(-5, 0, 2)' -l 'np.logspace(-5, 1, 2)' -bs 35 -de -act_func leaky_relu
exercise_c:
	python3 main.py -d Franke -p 1 -eta 'np.linspace(0.001,0.15,21)' -Ne 250 -act_func relu -history -push
	python3 main.py -d Franke -p 1 -eta 'np.linspace(0.001,0.15,21)' -Ne 250 -act_func leaky_relu -history -push
