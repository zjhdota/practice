from sklearn import datasets
from sklearn.svm import SVC

clf = SVC()
loaded_data = datasets.load_iris()

X, y = loaded_data.data, loaded_data.target

clf.fit(X, y)
"""
import pickle

with open('save/sample.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('save/sample.pkl', 'rb') as f:
    clf = pickle.load(f)

print(clf.predict(X))
"""


from sklearn.externals import joblib

joblib.dump(clf, 'save/joblib.pkl')

clf = joblib.load('save/joblib.pkl')

print(clf.predict(X))

