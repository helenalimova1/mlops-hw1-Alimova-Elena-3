
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib, os

df = pd.read_csv("data/train.csv")
X = df.drop("label", axis=1)
y = df["label"]

model = LogisticRegression(max_iter=300)
model.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
print("Saved models/model.pkl")
