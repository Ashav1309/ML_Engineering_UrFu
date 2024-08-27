import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Загрузка тестовых данных
test_data = pd.read_csv('data/test/temperature_test.csv')

# Загрузка обученной модели
model = joblib.load('trained_model.pkl')

# Предобработка тестовых данных (используем тот же StandardScaler)
scaler = StandardScaler()
scaled_temperature_test = scaler.fit_transform(test_data['Temperature'].values.reshape(-1, 1))

# Предсказание с использованием обученной модели
predictions = model.predict(scaled_temperature_test)

# Вывод результатов тестирования модели
print(predictions)