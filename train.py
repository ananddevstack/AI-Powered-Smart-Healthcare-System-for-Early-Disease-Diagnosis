import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Train Diabetes Model
print("Training Diabetes Model...")
df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
pickle.dump(model, open("diabetes_model.pkl", "wb"))
print("✅ Diabetes model saved as diabetes_model.pkl")

# Train Heart Disease Model
print("Training Heart Disease Model...")
df = pd.read_csv("heart.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
pickle.dump(model, open("heart_model.pkl", "wb"))
print("✅ Heart Disease model saved as heart_model.pkl")

# Train Cancer Risk Model
print("Training Cancer Model...")
df = pd.read_csv("cancer.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
pickle.dump(model, open("cancer_model.pkl", "wb"))
print("✅ Cancer model saved as cancer_model.pkl")
