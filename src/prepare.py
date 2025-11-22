
import pandas as pd
import os

# Загрузка данных
data_path = 'data/raw/iris.csv'
df = pd.read_csv(data_path, header=None)

# Подготовка данных
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Сохранение обработанных данных
pd.concat([X_train, y_train], axis=1).to_csv('data/processed/train.csv', index=False)
pd.concat([X_test, y_test], axis=1).to_csv('data/processed/test.csv', index=False)
