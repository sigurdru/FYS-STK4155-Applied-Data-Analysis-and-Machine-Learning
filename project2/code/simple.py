import utils
from NeuralNetwork import FFNN
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skd
from sklearn.model_selection import train_test_split as tts
import pickle

np.random.seed(2021)

class Args:
    dataset = "MNIST"

# file = "/home/hakon/Documents/Dataset/cifar-10-batches-py/data_batch_1"
# with open(file, "rb") as fo:
#     d = pickle.load(fo, encoding="bytes")
# imgs = d[b"data"]
# labels = np.asarray(d[b"labels"])

# imgs = imgs / 255

data = skd.load_digits()
# print(data)
imgs = data["data"]
labels = data["target"]

imgs /= imgs.max()

def plot_img(img):
    s = img.shape[1]
    c = 1
    l = int(np.sqrt(s // c))
    i = img.reshape(c, l, l)
    i = i.transpose(1, 2, 0)
    plt.imshow(i)


labels_onehot = utils.categorical(labels)


I_, I_test, T_, T_test = tts(imgs, labels_onehot, test_size=0.2)

NN = FFNN(I_, T_,
          hidden_nodes=[40, 20],
          batch_size=100,
          learning_rate=0.03,
          lmb=0,
          gamma=0.6,
          activation="sigmoid",
          cost="cross_entropy",
          output_activation="softmax",
          test_data=(I_test, T_test),
          )

th, _ = NN.train(100, True)
plt.plot(th)
plt.show()

probs = NN.predict(I_)
pred = np.argmax(probs, axis=1).reshape(-1, 1)
train = np.argmax(T_, axis=1).reshape(-1, 1)
print(np.sum(train == pred) / len(train))

probs = NN.predict(I_test)
pred = np.argmax(probs, axis=1).reshape(-1, 1)
test = np.argmax(T_test, axis=1).reshape(-1, 1)
print(np.sum(test == pred) / len(test))