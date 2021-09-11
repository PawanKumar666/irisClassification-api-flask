import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("Iris.csv")
data = data.drop(["Id"],axis=1)

encoder = LabelEncoder()
data["Species"] = encoder.fit_transform(data["Species"])

y = data["Species"]
train = data.drop(["Species"],axis=1)

x_train,x_test,y_train,y_test = train_test_split(train,y,test_size=0.2,random_state=42)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(x_train,y_train)

pickle.dump(clf, open('irisModel.pkl','wb'))

model = pickle.load(open('irisModel.pkl','rb'))

y_pred_ = model.predict([[6.7,3.0,5.4,2.3]])
print(y_pred_[0])