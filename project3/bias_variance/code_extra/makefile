LinearRegression:
	python3 bias_variance_main.py -m OLS -nr_c 12 -show
	python3 bias_variance_main.py -m Ridge -nr_c 12 -alpha 0.1 -show
	python3 bias_variance_main.py -m Lasso -nr_c 12 -alpha 0.1 -show
	python3 bias_variance_main.py -m Ridge -nr_c 14 -alpha 0.001 -plot_type regularization -show 
	python3 bias_variance_main.py -m Lasso -nr_c 14 -alpha 0.001 -plot_type regularization -show

NeuralNetwork:
	python3 bias_variance_main.py -m NN -nr_c 7

SVM:
	python3 bias_variance_main.py -m SVM -plot_type standard -kernel rbf -C 0.1 1 10 100 1000 -eps 0.2 -show
	python3 bias_variance_main.py -m SVM -plot_type 3d -kernel rbf -C 0.1 1 10 100 1000 -eps 0.01 0.1 1 10 100

all: LinearRegression NeuralNetwork SVM