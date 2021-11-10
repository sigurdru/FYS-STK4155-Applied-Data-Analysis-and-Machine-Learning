
NN_classification:
	python3 main.py -d Cancer -hn 20,20 -eta 'np.logspace(-5, 0, 11)' -l 'np.logspace(-5, 1, 11)' -bs 35 -de -act_func sigmoid
	python3 main.py -d Cancer -hn 20,20 -eta 'np.logspace(-5, 0, 11)' -l 'np.logspace(-5, 1, 11)' -bs 35 -de -act_func relu
	python3 main.py -d Cancer -hn 20,20 -eta 'np.logspace(-5, 0, 11)' -l 'np.logspace(-5, 1, 11)' -bs 35 -de -act_func leaky_relu
test:
	python3 main.py -d Cancer -hn 20,20 -eta 'np.logspace(-5, 0, 2)' -l 'np.logspace(-5, 1, 2)' -bs 35 -de -act_func leaky_relu
NN_regression:
	python3 main.py -d Franke -p 1 -eta 'np.logspace(-4, 0, 11)' -l 'np.logspace(-5, -1, 11)' -Ne 100
	python3 main.py -d Franke -p 1 -eta 'np.logspace(-4, 0, 11)' -l 'np.logspace(-5, -1, 11)' -de -Ne 100
	python3 main.py -d Franke -p 1 -eta 'np.logspace(-4, 0, 11)' -l 'np.logspace(-5, -1, 11)' -de -Ne 100 -ga 0.5
	python3 main.py -d Franke -p 1 -eta 'np.logspace(-4, 0, 11)' -l 'np.logspace(-5, -1, 11)' -de -Ne 100 -act_func relu
	python3 main.py -d Franke -p 1 -eta 'np.logspace(-4, 0, 11)' -l 'np.logspace(-5, -1, 11)' -de -Ne 100 -act_func leaky_relu