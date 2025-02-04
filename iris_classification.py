# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import load_iris

# Step 2: Load the Dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 3: Explore the Data
print("Feature Names:", iris.feature_names)
print("Target Names:", iris.target_names)
print("First 5 rows of the dataset:\n", pd.DataFrame(X, columns=iris.feature_names).head())

# Step 4: Preprocess the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Train a Machine Learning Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the Model
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Step 7: Make Predictions
new_flower = [[7, 3.2, 4.7, 1.4]]  # Example measurements
new_flower_scaled = scaler.transform(new_flower)
prediction = model.predict(new_flower_scaled)
print("Predicted species:", iris.target_names[prediction][0])