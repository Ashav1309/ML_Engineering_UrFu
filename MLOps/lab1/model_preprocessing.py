import pandas as pd
from sklearn.preprocessing import StandardScaler

# Загрузка данных
train_data = pd.read_csv('data/train/temperature_train.csv')

# Предобработка данных с использованием StandardScaler
scaler = StandardScaler()
scaled_temperature = scaler.fit_transform(train_data['Temperature'].values.reshape(-1, 1))

# Сохранение предобработанных данных
train_data['scaled_temperature'] = scaled_temperature
train_data.to_csv('data/train/preprocessed_train_data.csv', index=False)