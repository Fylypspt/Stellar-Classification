import pandas as pd

DATA_PATH = "data/stars.csv"
MODEL_PATH = "model/train1.pkl"
XTEST_PATH = "model/X_test.pkl"
YTEST_PATH = "model/y_test.pkl"

FEATURES = ["u", "g", "r", "i", "z", "redshift"]

TEST_SIZE = 0.2
SEED = 42
CONFIDENCE_THRESHOLD = 0.6

df = pd.read_csv(DATA_PATH)
X = df[FEATURES]
y = df["class"]

LOAD_MODEL = True