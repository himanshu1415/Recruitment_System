from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn import tree
df = pd.read_csv("PerpData.csv", header = 0)
def dec(per=60,back=1,intern=2,r=60,comm=60):
    features = list(df.columns[1:6])
    y = df["Hired"]
    x = df[features]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x,y)
    return clf.predict([[per,back,r,intern,comm]])
def randomf(per=60,back=1,intern=2,r=60,comm=60):
    features = list(df.columns[1:6])
    y = df["Hired"]
    x = df[features]
    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(x, y)
    return clf.predict([[per,back,intern,r,comm]])
    
