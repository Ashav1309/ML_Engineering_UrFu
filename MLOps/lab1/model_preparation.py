import pandas as pd
from sklearn.linear_model import LinearRegression

# Загрузка предобработанных данных
train_data = pd.read_csv('data/train/preprocessed_train_data.csv')

# Подготовка данных для обучения модели
X_train = train_data['scaled_temperature'].values.reshape(-1, 1)
y_train = train_data['Temperature'].values

# Обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Сохранение обученной модели
import joblib
joblib.dump(model, 'trained_model.pkl')