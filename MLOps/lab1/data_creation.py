import os
import numpy as np
import pandas as pd

# Создаем папки "train" и "test", если они еще не созданы
if not os.path.exists('data/train'):
    os.makedirs('data/train')
if not os.path.exists('data/test'):
    os.makedirs('data/test')

# Создаем данные для тренировочного набора
np.random.seed(42)
data_train = pd.date_range(start='1/1/2022', periods=100)
temperature_train = np.sin(np.arange(0, 10, 0.1)) * 20 + np.random.normal(0, 2, 100)
temperature_train[10:15] += 10  # Добавляем аномалии
df_train = pd.DataFrame({'Date': data_train, 'Temperature': temperature_train})
df_train.to_csv('data/train/temperature_train.csv', index=False)

# Создаем данные для тестового набора с добавлением аномалий
data_test = pd.date_range(start='1/1/2022', periods=50)
temperature_test = np.sin(np.arange(0, 5, 0.1)) * 20 + np.random.normal(0, 2, 50)
temperature_test[10:15] += 10  # Добавляем аномалии
df_test = pd.DataFrame({'Date': data_test, 'Temperature': temperature_test})
df_test.to_csv('data/test/temperature_test.csv', index=False)

print("Данные успешно созданы и сохранены в папкe 'data'.")