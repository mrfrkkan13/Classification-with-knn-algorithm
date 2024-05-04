import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV

# Load the datasets
data = pd.read_csv("C:/Dev/Python/TitanicDataSetProject/DataSet/train.csv")
test = pd.read_csv("C:/Dev/Python/TitanicDataSetProject/DataSet/test.csv")

# Feature extraction
data = data.drop(columns=["Name", "Ticket", "PassengerId", "Fare"])
test = test.drop(columns=["Name", "Ticket", "PassengerId", "Fare"])

data["FamCnt"] = data["SibSp"] + data["Parch"]
data = data.drop(columns=["SibSp", "Parch", "Cabin"])
test["FamCnt"] = test["SibSp"] + test["Parch"]
test = test.drop(columns=["SibSp", "Parch", "Cabin"])
# Label encoding and One-Hot encoding
le = LabelEncoder()
data["Sex"] = le.fit_transform(data["Sex"])
data = pd.get_dummies(data, columns=["Embarked", "Pclass"], dtype=int)

test["Sex"] = le.fit_transform(test["Sex"])
test = pd.get_dummies(test, columns=["Embarked", "Pclass"], dtype=int)


data = data.fillna(data.mean())
data["Age"] = data["Age"].astype(int)

test = test.fillna(test.mean())
test["Age"] = test["Age"].astype(int)


scaler = StandardScaler()
X_train = data.drop(columns=["Survived"])
y_train = data["Survived"]

X_train_scaled = scaler.fit_transform(X_train)
test_scaled = scaler.transform(test)

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = SVC()
clf = GridSearchCV(svc, parameters)
model = clf.fit(X_train_scaled, y_train)


predictions = model.predict(test_scaled)


ori_data = pd.read_csv("C:/Dev/Python/TitanicDataSetProject/DataSet/test.csv")
submission = pd.DataFrame({'PassengerId': ori_data['PassengerId'], 'Survived': predictions})
submission.to_csv("C:/Dev/Python/TitanicDataSetProject/DataSet/submission.csv", header=True, index=False)

from sklearn.metrics import accuracy_score

# Eğitim veri seti üzerinde tahmin yapma
y_train_pred = model.predict(X_train_scaled)

# Eğitim veri seti üzerindeki doğruluk değerini hesaplama
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Eğitim veri seti üzerindeki doğruluk değeri:", train_accuracy)
