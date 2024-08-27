from fastapi import FastAPI

from data import download_and_save_data
from entity import IrisRequest
from model import processing, check_result
from test_model import get_model, get_predict
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
   
app = FastAPI()
download_and_save_data() # Загрузка и сохранение данных
processing() # Обработка данных
check_result() # Проверка результатов
model = get_model() # Получение модели


@app.post("/predict")
def get_history(iris_request: IrisRequest):
    """Запрос истории ответов текущей сессии."""
    result = get_predict(model, iris_request) # Получение предсказаний с использованием модели
    return result
