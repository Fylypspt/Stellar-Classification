from src.config import *
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def load_or_train_model(load_existing):
    if load_existing:
        model = joblib.load(MODEL_PATH)
        X_test = joblib.load(XTEST_PATH)
        y_test = joblib.load(YTEST_PATH)
    else:
        
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=SEED,
            stratify=y
        )

        model = RandomForestClassifier(
            random_state=SEED,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_features=MAX_FEATURES
        )
        model.fit(X_train, y_train)

        # param_grid = {
        #     'max_depth': [5, 10, 20, None],
        #     'min_samples_leaf': [1, 2, 5, 10],
        #     'max_features': ['sqrt', 'log2']
        # }
        # grid_search = GridSearchCV(
        #     estimator=RandomForestClassifier(random_state=SEED),
        #     param_grid=param_grid,
        #     scoring='neg_log_loss', 
        #     cv=3, 
        #     n_jobs=-1
        # )
        # grid_search.fit(X_train, y_train)
        # model = grid_search.best_estimator_
        # print(f"The best settings to reduce surprise are: {grid_search.best_params_}")

        joblib.dump(model, MODEL_PATH)
        joblib.dump(X_test, XTEST_PATH)
        joblib.dump(y_test, YTEST_PATH)

    return model, X_test, y_test

def simulate_drift(X, noise_level=0.2):
    X_drifted = X.copy()
    for col in X.columns:
        X_drifted[col] += np.random.normal(
            0,
            X[col].std() * noise_level,
            size=len(X)
        )
    return X_drifted

def predict_star_class(model, features, data, threshold):
    sample = pd.DataFrame([data], columns=features)

    probs = model.predict_proba(sample)[0]
    max_prob = probs.max()
    pred_class = model.classes_[probs.argmax()]

    confidence_map = dict(zip(model.classes_, probs))

    if max_prob < threshold:
        return "UNCERTAIN", confidence_map

    return pred_class, confidence_map