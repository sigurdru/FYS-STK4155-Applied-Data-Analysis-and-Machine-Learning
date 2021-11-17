"""
Code to study the correlation matrix of breast cancer data
Not used in project
"""

import sklearn.datasets as skd
import seaborn as sns
import pandas as pd

data = skd.load_breast_cancer()
X, t = data.data, data.target.reshape(-1, 1)

print(data["target_names"])
df = pd.DataFrame(X, columns=data.feature_names)
df["target"] = t
corrmat = df.corr()

feature = "mean radius"

good_predictors = df.loc[:, lambda x: abs(corrmat[feature]) > 0.5]
good_predictors[feature] = df[feature]
good_predictors["tumor"] = t

print(df)
print(good_predictors)

sns.heatmap(df.corr(), annot=True)
sns.mpl.pyplot.show()

