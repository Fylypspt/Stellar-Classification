import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt

df = pd.read_csv("stars.csv")

features = ["u", "g", "r", "i", "z", "redshift"]
X = df[features] #Input
y = df["class"] #Predict

try:
    model = joblib.load("train1.pkl")
    X_test = joblib.load("X_test.pkl")
    y_test = joblib.load("y_test.pkl")
except FileNotFoundError:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    #Learned patterns from x and Y
    joblib.dump(model, "train1.pkl")
    #Data used to train the model
    joblib.dump(X_test, "X_test.pkl")
    joblib.dump(y_test, "y_test.pkl")

print(f"Accuracy: {model.score(X_test, y_test):.4%}")

def predict_star_class(model, features: list, new_object_data: list) -> tuple:
    new_object = pd.DataFrame([new_object_data], columns=features)

    prediction = model.predict(new_object)[0]
    probabilities = model.predict_proba(new_object)[0]

    confidence_dict = {cls: prob for cls, prob in zip(model.classes_, probabilities)}

    return prediction, confidence_dict

new_star_data = [19.2, 18.7, 18.3, 18.1, 18.0, 0.0001]

predicted_class, class_confidences = predict_star_class(model,features,new_star_data)

# Prints
print(f"Predicted Class: {predicted_class}")
print("Probability:")
for cls, prob in class_confidences.items():
    print(f"  {cls}: {prob:.4%}")


plt.figure(figsize=(8,6))

for cls in model.classes_:
    idx = y_test == cls
    plt.scatter(X_test.loc[idx, "u"], X_test.loc[idx, "redshift"], label=cls, alpha=0.45)

plt.xlabel("u")
plt.ylabel("redshift")
plt.title("Scatter plot of stars by class")
plt.legend()
plt.show()