import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("stars.csv")

features = ["u", "g", "r", "i", "z"]
X = df[features]
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

def predict_star_class(model, features: list, new_object_data: list) -> tuple:
    new_object = pd.DataFrame(
        [new_object_data],
        columns=features
    )

    prediction = model.predict(new_object)[0]
    probabilities = model.predict_proba(new_object)[0]

    confidence_dict = {
        cls: prob for cls, prob in zip(model.classes_, probabilities)
    }

    return prediction, confidence_dict

new_star_data = [19.2, 18.7, 18.3, 18.1, 18.0]

predicted_class, class_confidences = predict_star_class(model,features,new_star_data)

# Prints
print(f"Predicted Class: {predicted_class}")
print("Probability:")
for cls, prob in class_confidences.items():
    print(f"  {cls}: {prob:.4f}")