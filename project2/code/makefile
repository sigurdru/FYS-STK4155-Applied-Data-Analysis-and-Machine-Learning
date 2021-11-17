SGD_regression:
	python3 main.py -d Franke -m reg -Ne 150 -eta 'np.linspace(0.01,0.9,21)' 
	python3 main.py -d Franke -m reg -Ne 150 -eta 'np.linspace(0.01,0.5,21)'
	python3 main.py -d Franke -m reg -Ne 150 -eta 0.25 -bs 0 
	python3 main.py -d Franke -m reg -Ne 150 -eta 'np.linspace(0.1,0.5,11)' -l 'np.logspace(-5,0,11)'
	python3 main.py -d Franke -m reg -Ne 150 -eta 0.25 -ga -1
	python3 main.py -d Franke -m reg -Ne 150 -eta 'np.linspace(0.1,0.9,21)' -de 
NN_regression:
	python3 main.py -d Franke -p 1 -eta 'np.linspace(0.001, 0.9, 21)' -l 0 -Ne 150
	python3 main.py -d Franke -p 1 -eta 'np.linspace(0.05, 0.5, 21)' -l 0 -Ne 150
	python3 main.py -d Franke -p 1 -eta 'np.linspace(0.05, 0.5, 21)' -l 0 -Ne 250 -de
	python3 main.py -d Franke -p 1 -eta 'np.linspace(0.001, 0.15, 21)' -l 0 -Ne 250 -act_func relu
	python3 main.py -d Franke -p 1 -eta 'np.linspace(0.001, 0.15, 21)' -l 0 -Ne 250 -act_func leaky_relu
	python3 main.py -d Franke -p 1 -eta 'np.linspace(0.001, 0.15, 21)' -l 0 -Ne 250 -act_func sigmoid
	python3 main.py -d Franke -p 1 -eta 'np.linspace(0.05, 0.5, 11)' -l 'np.logspace(-7, 1, 15)' -Ne 100
NN_reg_layers:
	python3 main.py -d Franke -p 1 -eta 'np.linspace(0.001, 0.15, 11)' -act_func relu -history -l 0 -Ne 100 -hn 20
	python3 main.py -d Franke -p 1 -eta 'np.linspace(0.001, 0.15, 11)' -act_func relu -history -l 0 -Ne 100 -hn 10,10
	python3 main.py -d Franke -p 1 -eta 'np.linspace(0.001, 0.15, 11)' -act_func relu -history -l 0 -Ne 100 -hn 5,10,5
	python3 main.py -d Franke -p 1 -eta 'np.linspace(0.001, 0.15, 11)' -act_func relu -history -l 0 -Ne 100 -hn 5,5,5,5
NN_classification:
	python3 main.py -d Cancer -Ne 100 -hn 10,10 -eta 'np.logspace(-4, 0, 9)' -l 0 -history -bs 35 -act_func sigmoid
	python3 main.py -d Cancer -Ne 100 -hn 10,10 -eta 'np.logspace(-4, 0, 9)' -l 0 -history -bs 35 -de -act_func sigmoid
	python3 main.py -d Cancer -Ne 100 -hn 10,10 -eta 'np.logspace(-6, -1, 11)' -l 0 -history -bs 35 -de -act_func relu
	python3 main.py -d Cancer -Ne 100 -hn 10,10 -eta 'np.logspace(-6, -1, 11)' -l 0 -history -bs 35 -de -act_func leaky_relu
	python3 main.py -d Cancer -Ne 100 -hn 10,10 -eta 'np.logspace(-6, -1, 11)' -l 'np.logspace(-7,1,17)' -bs 35 -de -act_func leaky_relu
NN_class_layers:
	python3 main.py -d Cancer -Ne 50 -hn 20 -eta 'np.logspace(-6, -1, 11)' -l 0 -history -bs 35 -de -act_func relu
	python3 main.py -d Cancer -Ne 50 -hn 20,20 -eta 'np.logspace(-6, -1, 11)' -l 0 -history -bs 35 -de -act_func relu
	python3 main.py -d Cancer -Ne 50 -hn 20,20,20 -eta 'np.logspace(-6, -1, 11)' -l 0 -history -bs 35 -de -act_func relu
	python3 main.py -d Cancer -Ne 50 -hn 20,20,20,10 -eta 'np.logspace(-6, -1, 11)' -l 0 -history -bs 35 -de -act_func relu
Logistic_regression:
	python3 main.py -d Cancer -m reg -bs 5 -eta 'np.logspace(-5, -0.5, 11)' -Ne 200
	python3 main.py -d Cancer -m reg -bs 5 -eta 'np.logspace(-5, -1, 11)' -l 'np.logspace(-5, -3, 11)' -Ne 200

all: SGD_regression NN_regression NN_reg_layers NN_classification NN_class_layers Logistic_regression

