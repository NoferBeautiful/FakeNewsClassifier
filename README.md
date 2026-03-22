# FakeNewsClassifier

MLOps-система для классификации фейковых новостей на основе DistilBERT.

## Модели

| Модель | Описание |
|--------|----------|
| `baseline_frozen_encoder` | distilbert с замороженным энкодером, обучен на данных до середины 2017 |
| `baseline_frozen_encoder_tuned2018` | `baseline_frozen_encoder`, дообученный на 2018-2018 года |
| `baseline_frozen_encoder_trump` | `baseline_frozen_encoder`, дообученный на trump-данных |
| `baseline_unfrozen_encoder` | distilbert с размороженным энкодером, переобученный под данную задачу |

## Запуск

Сбор данных (создаёт train1, test1, train2, test2, train_trump, test_trump):

```bash
cd FakeNewsClassifier
python -m data_collection.collector
```

Обучение модели:

```bash
python run.py train --name v1
```

Дообучение модели:

```bash
python run.py update --name v2 --base-model v1
```

Валидация:

```bash
python run.py validate --model v1 --files data/processed/test1.csv --files data/processed/test2.csv
```

Инференс:

```bash
python run.py inference --model v1 --file data/processed/test1.csv
```

Предсказание одной новости:

```bash
python run.py predict --model v1 --text "Trump iran boom boom boom"
```

Интерпретация ответа:

```bash
python run.py explain --model v1 --text "Trump iran boom boom boom"
```

Сравнение 2 моделей:

```bash
python run.py compare --model1 v1 --model2 v2 --dataset data/processed/test1.csv
```

Сводка по моделям и метрикам:

```bash
python run.py summary
```

Эмуляция стриминга:

```bash
# Высокая нагрузка + большие батчи
python run.py stream --model baseline_frozen_encoder --file data/processed/test1.csv --rps 100 --batch-size 64 --limit 500

# Низкая нагрузка -> маленькие батчи с таймаутом
python run.py stream --model baseline_frozen_encoder --file data/processed/test2.csv --rps 2 --batch-size 8 --batch-timeout 3.0 --limit 50

# Обычный сценарий
python run.py stream --model baseline_frozen_encoder --file data/processed/test1.csv --rps 20 --batch-size 16 --report-interval 0.5 --limit 200

# С drift (переключение на trump-модель при >30% новостей с trump в батче)
python run.py stream --model baseline_frozen_encoder --file data/processed/test_trump.csv --trump-threshold 0.3 --batch-size 16 --limit 100
```

- `--rps` — запросы в секунду
- `--batch-size` — размер батча для инференса
- `--limit` — количество запросов
- `--report-interval` — периодичность отчета в секундах
- `--batch-timeout` — таймаут принудительной обработки батча, если он не заполнился
