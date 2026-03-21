# FakeNewsClassifier

MLOps-система для классификации фейковых новостей.

## Запуск

Сбор данных:

```bash
cd FakeNewsClassifier
python -m data_collection.collector
```

Обучение модели:

```bash
python train_bert.py --train-csv data/processed/train1.csv
```

Дообучение модели:

```bash
python update_bert.py --new-data-csv data/processed/train2.csv --model-dir models/bert/v_XXXXXX/final_model
```

---

## Критерии оценки

### Этап 1 — Сбор данных (3–10 баллов)

#### Обязательная часть

| Критерий | Баллы | Статус | Реализация |
|----------|-------|--------|------------|
| Разделение на батчи и эмуляция потока | 1 | Сделано | Трейн разделяется на батчи по месяцам и сплитится при необходимости на разные куски. Для эмуляции потока на инференсе появился столбец, обозначающий время прихода на сервер (по умолчанию в 1 RPS), по которому потом будет разделение на батчи |
| Хранилище сырых данных (файловая система) | 1 | Сделано | Батчи трейна сохраняются в `data/raw/`, обработанные данные в `data/processed/`. Пути настраиваются в [`config.yaml`](data_collection/config.yaml). Сплиты сохраняются в `data/processed/`. |
| Расчет метапараметров | 1-2 | Сделано | [`collector.py:calculate_meta()`](data_collection/collector.py) — для каждого датасета сохраняется JSON с метриками: `n_rows`, `n_fake`, `n_real`, `fake_ratio`, `text_len_mean/std/min/max`. Результат: `data/meta/*_meta.json` |

#### Дополнительные баллы

| Критерий | Баллы | Статус | Реализация |
|----------|-------|--------|------------|
| Конфигурационный файл | 1 | Сделано | [`config.yaml`](data_collection/config.yaml) — YAML-файл с параметрами: источники данных, частота батчей, пути, seed, rps, настройки splits |
| Интеграция с несколькими источниками | 2 | Сделано | [`collector.py:get_data_from_source()`](data_collection/collector.py) — поддержка `type: kaggle` (скачивание через kagglehub) и `type: local` (чтение из локальной папки). Источники объединяются в `load_all_sources()` |
| Логирование и обработка ошибок | 1-2 | Сделано | Используется модуль `logging`. Все ключевые операции логируются. Обработка ошибок через try/except в `get_data_from_source()` и `load_raw_data()` |
