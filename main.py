from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


CURRENT = Path(os.getcwd()).resolve()
file = CURRENT / "data/iris.data"
dataset = pd.read_csv(file, sep=",",names=["sepal_lenght", "sepal_width", "petal_lenght", "petal_width", "class"])

X = dataset.drop("class", axis=1)
y = dataset["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42, test_size=0.25)

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)
model = XGBClassifier()
model.fit(X_train, y_train_encoded)

y_pred = model.predict(X_test)

print("Accuracy : ", accuracy_score(y_test_encoded, y_pred))
print("Accuracy : ", confusion_matrix(y_test_encoded, y_pred))