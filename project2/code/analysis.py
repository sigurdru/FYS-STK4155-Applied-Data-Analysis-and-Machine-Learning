from collections import defaultdict
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.linear_model import LogisticRegression

# Our files
from NeuralNetwork import FFNN
import SGD
import plot
import utils


class NoneScaler(StandardScaler):
    """
    To have option of no scaling
    """

    def transform(self, x):
        return x


scale_conv = {"None": NoneScaler(),
                "S": StandardScaler(),
                "N": Normalizer(),
                "M": MinMaxScaler()}


def NN_regression(args):
    p = args.polynomial
    etas = args.eta
    lmbs = args.lmb
    scaler = scale_conv[args.scaling]
    if args.pred:
        # Reduce noise for surface plot comparison
        args.epsilon = 0.05
    x, y, z = utils.load_data(args)
    X = utils.create_X(x, y, p, intercept=False if p == 1 else True)
    X_train, X_test, z_train, z_test = utils.split_scale(X, z, args.tts, scaler)

    data = defaultdict(lambda: np.zeros((len(etas), len(lmbs)), dtype=float))
    MSE  = defaultdict(lambda: np.zeros((len(etas), args.num_epochs), dtype=float))
    R2   = defaultdict(lambda: np.zeros((len(etas), args.num_epochs), dtype=float))

    for i, eta in enumerate(etas):
        for j, lmb in enumerate(lmbs):
            np.random.seed(args.seed)
            NN = FFNN(X_train,
                      z_train,
                      args,
                      hidden_nodes=args.hidden_nodes,
                      batch_size=args.batch_size,
                      learning_rate=eta,
                      dynamic_eta=args.dynamic_eta,
                      lmb=lmb,
                      gamma=args.gamma,
                      wi=args.weight_initialization,
                      activation=args.act_func,
                      cost="MSE",
                      output_activation="none",
                      )

            NN.train(args.num_epochs, train_history=args.history, test=(X_test, z_test))

            # Rescale data to obtain correct values
            train_pred = utils.rescale_data(NN.predict(X_train), z)
            test_pred = utils.rescale_data(NN.predict(X_test), z)

            z_train_ = utils.rescale_data(z_train, z)
            z_test_ = utils.rescale_data(z_test, z)

            tr_mse = utils.MSE(z_train_, train_pred)
            te_mse = utils.MSE(z_test_, test_pred)

            print('mse train  : ', tr_mse)
            print('mse test   : ', te_mse)

            data["train MSE"][i][j] = tr_mse if tr_mse < 1 else np.nan
            data["test MSE"][i][j] = te_mse if te_mse < 1 else np.nan

            if args.history:
                # MSE and R2 at each epoch
                MSE["train MSE"][i] = NN.history["train_mse"]
                MSE["test MSE"][i]  = NN.history["test_mse"]
                R2["train R2"][i]   = NN.history["train_R2"]
                R2["test R2"][i]    = NN.history["test_R2"]

            if args.pred:
                """
                Plot fitted surface with original data
                """
                X_ = utils.split_scale(X, z, 0, scaler)[0] # Scale design matrix

                # Calculate output. Rescale values to the original
                z_pred = utils.rescale_data(NN.predict(X_), z)
                
                # Plot result of fit and exit
                plot.surface_fit(z_pred, z, x, y, args)

    print("\n"*3)
    print(f"Best NN train prediction: {(train:=data['train MSE'])[(mn:=np.unravel_index(np.nanargmin(train), train.shape))]} for eta = {etas[mn[0]]}, lambda = {lmbs[mn[1]]}")
    print(f"Best NN test prediction: {(test:=data['test MSE'])[(mn:=np.unravel_index(np.nanargmin(test), test.shape))]} for eta = {etas[mn[0]]}, lambda = {lmbs[mn[1]]}")
    if args.history:
        # Plot MSE for different etas as a function of epochs  
        # Turn following on and off as desired
        # plot.eta_epochs(MSE, args)
        plot.eta_epochs(MSE, args, vmax=0.07)
        # plot.eta_epochs(R2, args, vmin=0.55)

    else:
        plot.eta_lambda(data, args)


def linear_regression(args):
    """ SGD """
    print("Doing linear regression")
    p = args.polynomial
    etas = args.eta
    lmbs = args.lmb

    if args.pred:
        # Reduce noise for surface plot comparison
        args.epsilon = 0.05

    if args.scaling == "S":
        # Don't divide data by std for Franke 
        # Since std<1, causing too much increase of data values  
        scaler = StandardScaler(with_std=False)
    else:
        scaler = scale_conv[args.scaling]

    x, y, z = utils.load_data(args)
    X = utils.create_X(x, y, p, intercept=False if p == 1 else True)

    X_train, X_test, z_train, z_test = utils.split_scale(X, z, args.tts, scaler)
    beta0 = np.random.randn(utils.get_features(p))  # use same beta0 in all runs for comparisonability

    if args.gamma >= 0:
        # Analyze SGD 
        # Set gamma negative to study momentum
        if args.batch_size == 0:
            batch_sizes = np.array([720, 360, 240, 144, 72, 48, 36, 30, 24])
        else:
            batch_sizes = np.array([args.batch_size])
        args.batch_size = batch_sizes

        n_minibatches = X_train.shape[0] // batch_sizes
        args.nmb = n_minibatches

        data = defaultdict(lambda: np.zeros((len(etas),
                                            len(lmbs),
                                            len(batch_sizes),
                                            args.num_epochs), dtype=float))


        for i, eta in enumerate(etas):
            for j, lmb in enumerate(lmbs):
                for k, batch_size in enumerate(batch_sizes):
                    # np.random.seed(args.seed)
                    MSE_train, MSE_test, beta = SGD.SGD((X_train, X_test),
                                                    (z_train, z_test),
                                                    args,
                                                    beta0,
                                                    eta,
                                                    batch_size,
                                                    lmb,
                                                    args.gamma)

                    data["Train MSE"][i][j][k] = MSE_train
                    data["Test MSE"][i][j][k] = MSE_test

                    if args.pred:
                        """
                        Plot fitted prediction vs data set with epsilon=0.05
                        Good result:
                         - Ne  : 10 000 (lol)
                         - eta : 0.25 (to be adjusted)
                         - ga  : 0.5 
                        """
                        # Scale the full design matrix 
                        X_ = utils.split_scale(X, z, 0, scaler)[0] # Scale design matrix

                        # Calculate prediction. Rescale values 
                        z_pred = X_ @ beta + np.mean(z)

                        # Plot result of fit, and exit
                        plot.surface_fit(z_pred, z, x, y, args)

        print(f"Best NN train prediction: {(train:=data['Train MSE'])[(mn:=np.unravel_index(np.nanargmin(train), train.shape))]} for eta = {etas[mn[0]]}, lambda = {lmbs[mn[1]]}")
        print(f"Best NN test prediction: {(test:=data['Test MSE'])[(mn:=np.unravel_index(np.nanargmin(test), test.shape))]} for eta = {etas[mn[0]]}, lambda = {lmbs[mn[1]]}")
        plot.parameter_based(data, args)

    else:
        gammas = np.linspace(0, 0.70, 15)
        args.gamma = gammas
        data = defaultdict(lambda: np.zeros((len(gammas), args.num_epochs)))

        for i, gamma in enumerate(gammas):

            mom_MSE_train, mom_MSE_test, beta = SGD.SGD((X_train, X_test),
                                                (z_train, z_test),
                                                args,
                                                beta0,
                                                args.eta[0],
                                                args.batch_size,
                                                args.lmb[0],
                                                gamma=gamma)

            data["Train MSE"][i] = mom_MSE_train
            data["Test MSE"][i] = mom_MSE_test
        print(f"Best SGD test prediction: {(test:=data['Test MSE'])[(mn:=np.unravel_index(np.nanargmin(test), test.shape))]} for gamma = {gammas[mn[0]]}")
        plot.momentum(data, args)


def logistic_regression(args):
    etas = args.eta
    lmbs = args.lmb
    batch_size = args.batch_size

    #import dataset
    dataset = utils.load_data(args)
    z = dataset.target.reshape(-1, 1)
    X = dataset.data
    n_features = np.shape(X)[1]

    #split and scale data
    scaler = scale_conv[args.scaling]
    X_train, X_test, z_train, z_test = utils.split_scale(X, z, args.tts, scaler)
    #for easier updates of weights
    n_test_patients = np.shape(X_test)[0]
    n_train_patients = np.shape(X_train)[0]
    X_train = np.c_[X_train, np.ones((n_train_patients, 1))]
    X_test = np.c_[X_test, np.ones((n_test_patients, 1))]
    #initialize weights and bias
    W = np.random.randn(n_features + 1,1)
    #We want to see the prediction and accuracy of ScikitLearn
    print('-------------------------------')
    print('|Prediction using Scikit Learn|')
    SKLR = LogisticRegression(penalty='l2').fit(X_train, np.ravel(z_train))
    SK_score = SKLR.score(X_test, np.ravel(z_test))
    print(f'|Accuracy score: {SK_score:.11f}|')
    print('-------------------------------')

    data = defaultdict(lambda: np.zeros((len(etas), len(lmbs), args.num_epochs), dtype=float))
    for i, eta in enumerate(etas):
        for j, lmb in enumerate(lmbs):
            np.random.seed(args.seed)
            accuracy_train, accuracy_test, beta = SGD.SGDL((X_train, X_test),
                                                (z_train, z_test),
                                                W,
                                                args,
                                                eta,
                                                batch_size,
                                                lmb,
                                                args.gamma)
            data["Train accuracy"][i][j] = accuracy_train
            data["Test accuracy"][i][j] = accuracy_test
    if np.size(args.lmb) == 1 and np.size(args.eta) != 1:
        plot.plot_logistic_regression_epochs(data, args)
    elif np.size(args.lmb) != 1 and np.size(args.eta) != 1:
        plot.plot_logistic_regression(data, args)
    else:
        print('Train accuracy: ')
        print(data["Train accuracy"][-1][-1][-1])
        print('Test accuracy: ')
        print(data["Test accuracy"][-1][-1][-1])


def NN_classification(args):
    etas = args.eta
    lmbs = args.lmb

    dataset = utils.load_data(args)
    z_ = dataset.target.reshape(-1, 1)
    X = dataset.data
    
    z = utils.categorical(z_)
    scaler = scale_conv[args.scaling]
    X_train, X_test, z_train, z_test = utils.split_scale(X, z, args.tts, scaler)
    print(X_test.shape)
    data = defaultdict(lambda: np.zeros((len(etas), len(lmbs))))
    train_data = defaultdict(lambda: np.zeros((len(etas), args.num_epochs)))

    for i, eta in enumerate(etas):
        for j, lmb in enumerate(lmbs):
            np.random.seed(args.seed)

            NN = FFNN(X_train,
                      z_train,
                      args,
                      hidden_nodes=args.hidden_nodes,
                      batch_size=args.batch_size,
                      learning_rate=eta,
                      dynamic_eta=args.dynamic_eta,
                      lmb=lmb,
                      gamma=args.gamma,
                      wi=args.weight_initialization,
                      activation=args.act_func,
                      cost="cross_entropy",
                      output_activation="softmax",
                      )
            NN.train(args.num_epochs, train_history=args.history, test=(X_test, z_test))
            data["train accuracy"][i][j] = NN.predict_accuracy(X_train, z_train)
            data["test accuracy"][i][j] = NN.predict_accuracy(X_test, z_test)

            train_data["train accuracy"][i] = NN.history["train_accuracy"]
            train_data["test accuracy"][i] = NN.history["test_accuracy"]
            print("Test accuracy: ", data["test accuracy"][i][j])
            print()

            if args.history:
                if len(etas) == 1:
                    plot.train_history(NN, args)
                    exit()

    print(f"Best NN train prediction: {(train:=data['train accuracy'])[(mn:=np.unravel_index(np.nanargmax(train), train.shape))]} for eta = {etas[mn[0]]}, lambda = {lmbs[mn[1]]}")
    print(f"Best NN test prediction: {(test:=data['test accuracy'])[(mn:=np.unravel_index(np.nanargmax(test), test.shape))]} for eta = {etas[mn[0]]}, lambda = {lmbs[mn[1]]}")
    
    if args.history:
        plot.eta_epochs(train_data, args, vmax=1, vmin=0)
    else:
        plot.eta_lambda(data, args, NN=True)
