import sklearn.datasets as skd
import seaborn as sns
import pandas as pd

data = skd.load_breast_cancer()
X, t = data.data, data.target.reshape(-1, 1)

df = pd.DataFrame(X, columns=data.feature_names)
corrmat = df.corr()

good_predictors = df.loc[:, lambda x: abs(corrmat["mean radius"]) > 0.5]

print(df)
print(good_predictors)

sns.heatmap(df.corr(), annot=True)
sns.mpl.pyplot.show()

