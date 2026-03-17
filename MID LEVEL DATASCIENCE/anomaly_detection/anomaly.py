import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("transactions.csv")

# Features
X = data[["Amount","Time"]]

# Model
model = IsolationForest(contamination=0.2)
data["Anomaly"] = model.fit_predict(X)

# Convert output (-1 = anomaly)
data["Anomaly"] = data["Anomaly"].map({1:"Normal",-1:"Anomaly"})

print(data)

# Visualization
colors = ["red" if x=="Anomaly" else "blue" for x in data["Anomaly"]]

plt.scatter(data["Time"], data["Amount"], c=colors)
plt.xlabel("Time")
plt.ylabel("Amount")
plt.title("Transaction Anomaly Detection")
plt.show()