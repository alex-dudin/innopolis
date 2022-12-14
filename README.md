
# Решение задачи классификации агрокультур (hacks-ai.ru)

## Получение высот над уровнем моря для всех полей

В модели используется признак "высота над уровнем моря".

Значения для train и test данных приведены в файлах:

- data/train_elevation.csv

- data/test_elevation.csv

Для получения этих высот использовалась библиотека PyGMT (https://www.pygmt.org/).
Рекомендуется использовать `conda` для установки этой библиотеки, так как она тянет за
собой большое количество зависимостей.

Далее необходимо запустить скрипт `elevation.py`. Пример:
```
python elevation.py -i path/to/train.csv -o data/train_elevation.csv
python elevation.py -i path/to/test.csv -o data/test_elevation.csv
```

Эти скрипты нужно будет запустить несколько раз. Каждый запуск будет обрабатываться по 25 регионов `1гр x 1гр`. При большем количество регионов, PyGMT вылетает с ошибкой.

## Обучение моделей и получение предсказаний

Для обучения моделей LightGBM используется ноутбук `innopolis_lightgbm.ipynb`. В ходе соревнования он запускался в Google Colab на стандартной машине без GPU.

Для обучения моделей TabNet используется ноутбук `innopolis_tabnet.ipynb`. В ходе соревнования он запускался на Kaggle на машине с GPU.

Для LightGBM необходимо выполнить код из ноутбука 3 раза:

- с параметрами по умолчанию (используются все столбцы `nd_mean_*`, параметр `colsample_bytree` не задан)

- без 8 столбцов `nd_mean_*` (`nd_mean_2021-04-20`, `nd_mean_2021-04-22`, `nd_mean_2021-04-23`, `nd_mean_2021-05-09`, `nd_mean_2021-06-22`, `nd_mean_2021-06-25`, `nd_mean_2021-07-08` и `nd_mean_2021-08-27`)

- без 8 столбцов `nd_mean_*` и с параметром `colsample_bytree=0.5`

Для TabNet необходимо выполнить код из ноутбука 2 раза:

- с параметрами по умолчанию (используются все столбцы `nd_mean_*`)

- без 8 столбцов `nd_mean_*` (список столбцов приведен выше)

Примеры предиктов есть в папке `data/blend`.

## Получение финального решения

Для получения финального решения, необходимо усреднить предсказания от моделей `lightgbm` и `tabnet`.
Для этого используется скрипт `create_submission.py`.

Пример:
```
python create_submission.py -i data\blend\tabnet89.csv -i data\blend\tabnet93.csv -i data\blend\lightgbm88.csv -i data\blend\lightgbm90.csv -i data\blend\lightgbm92.csv -o submission.csv
```
