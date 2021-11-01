import numpy as np
from NeuralNetwork import FFNN
from analysis import split_scale, NoneScaler
np.random.seed(2021)


def MSE(y, y_):
    return sum((y - y_) ** 2) / len(y)


new_X = lambda x: np.c_[x,]
f = lambda x: x ** 2

N = 100
ts = 0.2

x = np.random.rand(N, 1) * 2 - 1
y = f(x)

scaler = NoneScaler()

X = new_X(x)
x_train, x_test, y_train, y_test = split_scale(X, y, ts, scaler)

ols = np.linalg.pinv(x_train.T @ x_train) @ x_train.T @ y_train
ols_pred = x_test @ ols
NN = FFNN(x_train,
          y_train,
          hidden_nodes=[30, 10],
          batch_size=10,
          learning_rate=0.1,
          lmb=1e-3,
          gamma=0.05,
          )
NN.train(500)
pred = NN.predict(x_test)

print("OLS_MSE: ", MSE(y_test, ols_pred))
print("NN_MSE: ", MSE(y_test, pred))

do = lambda x: (y:=f(x), y_:=NN.predict(new_X(x)), 0.5 * (y - y_)**2)
# embed()


import matplotlib.pyplot as plt
x = np.linspace(-3,3,101)
y, yp, _ = do(x)
plt.plot(x, y, label="expected")
plt.plot(x, yp, label="Prediction")
plt.legend()
plt.show()

