import pandas as pd
import os

# URL для загрузки данных
csv_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# Путь для сохранения загруженного файла
cur_path = "./iris.csv"

# Названия столбцов для набора данных Iris
col_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']

# Функция для загрузки и сохранения данных Iris
def download_and_save_data():
    # Загрузка данных в DataFrame и сохранение в CSV-файл
    iris = pd.read_csv(csv_url, names=col_names)
    iris.to_csv(cur_path, index=False)

    # Проверка результата сохранения
    if os.path.exists(cur_path):
        print("Successful downloading")
    else:
        raise FileNotFoundError("Something wrong with IRIS dataset downloading")