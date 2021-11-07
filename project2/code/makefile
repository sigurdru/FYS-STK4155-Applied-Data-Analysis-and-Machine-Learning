
NN_classification:
	python3 main.py -d Cancer -hn 20,20 -eta 'np.logspace(-5, 0, 11)' -l 'np.logspace(-5, 1, 11)' -bs 35 -de -act_func sigmoid
	python3 main.py -d Cancer -hn 20,20 -eta 'np.logspace(-5, 0, 11)' -l 'np.logspace(-5, 1, 11)' -bs 35 -de -act_func relu
	python3 main.py -d Cancer -hn 20,20 -eta 'np.logspace(-5, 0, 11)' -l 'np.logspace(-5, 1, 11)' -bs 35 -de -act_func leaky_relu
test:
	python3 main.py -d Cancer -hn 20,20 -eta 'np.logspace(-5, 0, 2)' -l 'np.logspace(-5, 1, 2)' -bs 35 -de -act_func leaky_relu