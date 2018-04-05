# -*- encoding=utf-8 -*-
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import pydotplus
import os

os.environ["PATH"] += os.pathsep + r'D:\Graphviz2.38\bin'

np.random.seed(0)
# data = np.loadtxt('watermelon_2.csv', delimiter=',', encoding='utf-8')
# X = data[:, 1:7]
# y = data[:, 7]
with open('watermelon_2.csv', 'r', encoding='utf-8') as f:
    df = pd.read_csv(f)

# arr = LabelBinarizer().fit_transform(df['色泽'])
# print(arr)

df = pd.get_dummies(df)


index = []
[index.append(i) for i in np.random.randint(1, 17, 11) if i not in index]

df.index = df['编号']
df = df.drop('编号', axis=1)
df_train = df.iloc[index]
df_test = df.drop(index)

X = df.values[:, :-2]
y = df.values[:, -2:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

dot_data = tree.export_graphviz(model, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("watermelon.pdf")

