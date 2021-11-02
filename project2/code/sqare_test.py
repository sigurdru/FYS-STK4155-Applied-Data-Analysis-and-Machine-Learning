import matplotlib.pyplot as plt
import numpy as np
from NeuralNetwork import FFNN
from utils import split_scale
from analysis import NoneScaler
np.random.seed(2021)


def MSE(y, y_):
    return sum((y - y_) ** 2) / len(y)

f = lambda x: np.sin(x)

N = 1000

x = np.random.rand(N, 1) * 2 - 1
x *= 2 * np.pi
y = f(x)


NN = FFNN(x,
          y,
          hidden_nodes=[50, 20, 20],
          batch_size=100,
          learning_rate=0.25,
          lmb=1e-4,
          gamma=0.9,
          activation="sigmoid",
          )
NN.train(500)
pred = NN.predict(x)

print("train MSE: ", MSE(y, pred))

x_test = np.linspace(-1, 1, 1001) * 2 * max(x)
x_test = x_test
y_test = f(x_test)
yp = NN.predict(x_test[:, None])

plt.plot(x_test, y_test, label="expected")
plt.plot(x_test, yp, label="Prediction")
plt.plot([min(x), min(x)], [min(min(yp), min(y_test)), max(max(yp), max(y_test))], "r--", label="train domain")
plt.plot([max(x), max(x)], [min(min(yp), min(y_test)), max(max(yp), max(y_test))], "r--",)
plt.plot()
plt.legend()
plt.show()

