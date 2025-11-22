
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import joblib

# Загрузка данных
train_data = pd.read_csv('data/processed/train.csv')
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

# Логирование с MLflow
mlflow.start_run()
mlflow.log_param("model", "RandomForest")

# Обучение модели
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Оценка модели
test_data = pd.read_csv('data/processed/test.csv')
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
mlflow.log_metric("accuracy", acc)

# Сохранение модели
joblib.dump(model, 'model.pkl')
mlflow.log_artifact("model.pkl")
mlflow.end_run()
