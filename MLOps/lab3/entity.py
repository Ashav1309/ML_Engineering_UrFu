from pydantic import BaseModel

# Определение модели запроса для предсказания Iris
class IrisRequest(BaseModel):
    iris_entities: list

# Класс для представления сущности Iris, которая будет использоваться для предсказаний
class IrisPredictEntity:
    def __init__(self,
                 sepal_len: float,
                 sepal_wid: float,
                 petal_len: float,
                 petal_wid: float,
                 iris_type: str):
        self.sepal_len = sepal_len  # Длина чашелистика
        self.sepal_wid = sepal_wid  # Ширина чашелистика
        self.petal_len = petal_len  # Длина лепестка
        self.petal_wid = petal_wid  # Ширина лепестка
        self.iris_type = iris_type # Тип Iris (например, "setosa", "versicolor", "virginica")

# Класс для представления ответа на предсказание Iris
class IrisPredictResponse:
    def __init__(self, predictions: list):
        self.predictions = predictions