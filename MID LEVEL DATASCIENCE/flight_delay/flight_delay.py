import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("flights.csv")

# Encode categorical data
le = LabelEncoder()
data["Weather"] = le.fit_transform(data["Weather"])

# Features & target
X = data[["Departure_Time","Weather"]]
y = data["Delay"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# Model
model = RandomForestClassifier()
model.fit(X_train,y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test,y_pred))