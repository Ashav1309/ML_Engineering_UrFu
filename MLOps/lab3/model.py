import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import pickle
import os

# Путь для сохранения модели
model_path = "./model.pkl"

# Путь к файлу данных
data_path = "./iris.csv"


def processing():
    # Загрзука данных, отделение признаков от целевой переменной, разделение данных на обучающий и тестовый наборы
    iris = pd.read_csv(data_path)
    X = iris.drop("Species", axis=1)
    y = iris["Species"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    # Инициализация объекта для обработки данных
    scaler = StandardScaler()

    # Инициализация модели логистической регрессии
    model = LogisticRegression()

    # PIPELINE
    pipe = Pipeline(steps=[("scaler", scaler), ("logistic", model)])
    pipe.fit(X_train, y_train)

    # Получение предсказаний на тестовом наборе данных
    y_predict = pipe.predict(X_test)
    print(f"Metrics:\n {classification_report(y_test, y_predict)}")

    # Сохранение обученной модели в файл
    pickle.dump(model, open(model_path, 'wb'))

# Проверка наличия файла с сохраненной моделью
def check_result():
    if os.path.exists(model_path):
        print("Successful model saving")
    else:
        raise FileNotFoundError("Something wrong with model saving")

    # Удаление файла с данными
    os.remove(data_path)

    # Проверка успешного удаления файла с данными
    if not os.path.exists(data_path):
        print("Successful data removing")
    else:
        raise FileNotFoundError("Something wrong with data removing")