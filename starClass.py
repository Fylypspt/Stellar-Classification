import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from src.config import *
import src.func as funcs
import src.plot as graph

#Init
try:
    model, X_test, y_test = funcs.load_or_train_model(LOAD_MODEL)
except FileNotFoundError:
    model, X_test, y_test = funcs.load_or_train_model(False)

#Scores
y_pred = model.predict(X_test)

print(f"Accuracy: {model.score(X_test, y_test):.4%}")

X_drifted = funcs.simulate_drift(X_test)
print(f"Accuracy after drift: {model.score(X_drifted, y_test):.4%}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#Test
new_star = [20.0, 20.0, 20.0, 20.0, 20.0, 0.35]

predicted, probs = funcs.predict_star_class(
    model,
    FEATURES,
    new_star,
    CONFIDENCE_THRESHOLD
)

print(f"\nPredicted Class: {predicted}")
for cls, prob in probs.items():
    print(f"  {cls}: {prob:.4%}")

graph.graph(y_pred, y_test, model, X_test)