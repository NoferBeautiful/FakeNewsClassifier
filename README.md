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
python run.py train --name baseline_frozen_encoder
```

Дообучение модели:

```bash
python run.py update --name baseline_frozen_encoder_trump --base-model baseline_frozen_encoder
```

Валидация:

```bash
python run.py validate --model baseline_frozen_encoder --files data/processed/test1.csv --files data/processed/test2.csv
```

Инференс:

```bash
python run.py inference --model baseline_frozen_encoder --file data/processed/test1.csv
```

Предсказание одной новости:

```bash
python run.py predict --model baseline_frozen_encoder --text "Trump iran boom boom boom"
```

Интерпретация ответа:

```bash
python run.py explain --model baseline_frozen_encoder --text "Trump iran boom boom boom"
```

Сравнение 2 моделей:

```bash
python run.py compare --model1 baseline_frozen_encoder --model2 baseline_frozen_encoder_trump --dataset data/processed/test1.csv
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

# С пустыми запросами (20% запросов будут пустыми)
python run.py stream --model baseline_frozen_encoder --file data/processed/test1.csv --empty-ratio 0.2 --batch-size 16 --limit 100
```

- `--rps` — запросы в секунду
- `--batch-size` — размер батча для инференса
- `--limit` — количество запросов
- `--report-interval` — периодичность отчета в секундах
- `--batch-timeout` — таймаут принудительной обработки батча, если он не заполнился
- `--empty-ratio` — доля пустых запросов (0-1)

## HF Space

Демо: https://huggingface.co/spaces/Nofer/FakeNewsClassifier_mlops

## Этап 1 — Сбор данных (3–10 баллов)

### Обязательная часть

| Критерий | Баллы | Статус | Реализация |
|----------|-------|--------|------------|
| Функционал сбора потоковых данных: разделение исходного набора на батчи и эмуляция потока | 1 | Сделано | `model/streaming.py` — `StreamEmulator` эмулирует поток запросов с заданным RPS, батчирование через очередь с `batch_size` и `batch_timeout`, многопоточная архитектура (producer/consumer/reporter) |
| Разработка хранилища сырых данных: файловая система (1 балл) или БД (2 балла) | 1/2 | Сделано на 1 | Файловая система: `data/raw/` — сырые батчи по месяцам, `data/processed/` — train/test splits, `data/meta/` — метаданные JSON для каждого split |
| Расчет метапараметров | 1-2 | Сделано | `calculate_meta()` — сохраняет в JSON: `n_rows`, `n_fake`, `n_real`, `fake_ratio`, `text_len_mean/std/min/max`, `created_at` |

### Дополнительные баллы

| Критерий | Баллы | Статус | Реализация |
|----------|-------|--------|------------|
| Создание конфигурационного файла с гиперпараметрами сбора | 1 | Сделано | `data_collection/config.yaml` — параметры: `sources`, `raw_path`, `processed_path`, `batch_freq`, `seed`, `rps`, `splits` |
| Интеграция с несколькими источниками данных | 2 | Сделано на 1/2 | Поддержка двух типов источников в `config.yaml`: `type: kaggle` (загрузка через kagglehub) и `type: local` (локальные файлы). Метод `load_all_sources()` объединяет данные из всех источников |
| Система логирования и обработки ошибок при сборе данных | 1-2 | Сделано | `logging` модуль с уровнями INFO/WARNING/ERROR, try-except блоки с информативными сообщениями, валидация входных данных |

## Этап 2 — Анализ данных (3–13 баллов)

### Обязательная часть

| Критерий | Баллы | Статус | Реализация |
|----------|-------|--------|------------|
| Оценка и хранение показателей качества данных (data quality) | 1-2 | Сделано | `assess_data_quality()` — вычисляет и сохраняет в JSON (`reports/data_quality.json`) метрики: пропуски (абсолютные + %), дубликаты (полные и по text), распределение label (метка фейк/правдивая новость), статистики длин текста/заголовков (mean, std, min, max, медиана, квантили), распределение по subject(темы статьи), распределение по source (источник, из какого датасета данный текст) временной диапазон, число пустых/коротких текстов |
| Применение методов построения ассоциативных правил (Apriori / FP-tree) | 1-3 | Сделано | `mine_association_rules()` — бинаризация 17 признаков (`is_fake`, `is_real`, `has_trump`, `has_aggression`, `is_short/long/medium_text`, `subject_*`, `source_*` и др.) через `create_binary_features()`. Применяется FP-Growth (mlxtend), генерируются правила с фильтрацией по confidence и lift. Автоматический выбор 5+ разнообразных правил через `_select_interesting_rules()`. Результат сохраняется в: `reports/association_rules.json` |
| Базовая очистка данных на основе порогов допустимых значений качества | 1 | Сделано | `clean_data()` — 6 шагов очистки с настраиваемыми порогами (`DEFAULT_CLEANING_THRESHOLDS`): удаление пустых text, невалидных label, фильтр по длине текста (min 50 / max 50000 символов), фильтр по длине заголовка, дедубликация по text, опциональное удаление невалидных дат. Результат сохраняется в: `reports/cleaning_report.json` |

### Дополнительные баллы

| Критерий | Баллы | Статус | Реализация |
|----------|-------|--------|------------|
| Автоматический EDA | 1-2 | Сделано | `run_automatic_eda()` — генерирует два отчёта: (1) `reports/eda_sweetviz.html` — сравнительный HTML-отчёт Fake vs Real через sweetviz; (2) `reports/eda_plots.pdf` — PDF (matplotlib/seaborn) с 7 типами графиков: распределение label (фейк/не фейк новость), категориальные признаки по классам, гистограммы числовых фичей, корреляционная матрица, boxplot топ-12 фичей, бинарные признаки по label. Автоматический семплинг для больших датасетов |
| Добавление Feature Engineering | 1-2 | Сделано | `add_features()` — создаёт 33 новых признака в 5 группах: (1) текстовые статистики (`text_len_chars/words`, `text_n_sentences`, `text_avg_word/sentence_len`, `title_len_*`, `title_to_text_ratio`); (2) стилистические маркеры (`upper_ratio`, `caps_word_count/ratio`, `exclamation/question/ellipsis/quote_count`, `punct_count/ratio`); (3) контентные бинарные (`has_trump`, `has_aggression`, `has_url`, `has_email`, `has_number`, `has_breaking`, `has_exclusive`, `has_shocking`); (4) лексическое разнообразие (`type_token_ratio`, `hapax_ratio`); (5) временные (`day_of_week`, `month`, `year`, `is_weekend`) |
| Генерация отчетов о качестве данных | 1 | Сделано | `generate_quality_report()` — сводный HTML-отчёт (`reports/data_quality_report.html`), объединяющий результаты assess_data_quality, clean_data и текущее состояние данных: summary-метрики, таблица пропусков с цветовой индикацией, распределение label и subject, таблица шагов очистки, выбросы по IQR, автоматические рекомендации по улучшению качества |

## Этап 3 — Подготовка данных (входит в pipeline построения модели) (0–5 баллов)

### Обязательная часть

| Критерий | Баллы | Статус | Реализация |
|----------|-------|--------|------------|
| Обработка пропусков | 0-1 | Сделано | `handle_missing_values()` — числовые: заполнение median/mean/zero (настраивается), бинарные: заполнение 0, категориальные: заполнение "unknown". Опционально добавляет флаги *_was_nan. Логирование каждого заполнения |
| Обработка категориальных переменных | 0-1 | Сделано | `encode_categorical()` — три метода: One-Hot Encoding (drop_first=True), Label Encoding, Frequency Encoding. Применяется к колонке `subject` |
| Обработка числовых переменных | 0-1 | Сделано | `scale_numeric()` — четыре метода: StandardScaler, MinMaxScaler, RobustScaler, None. Бинарные признаки автоматически исключаются из масштабирования. Объединяющий pipeline: `prepare_data()` |

### Дополнительные баллы

| Критерий | Баллы | Статус | Реализация |
|----------|-------|--------|------------|
| Создание нескольких вариантов предобработки с дальнейшим перебором при поиске лучшей модели | 1-2 | Сделано | `create_text_preprocessing_variants()` — создаёт 5 вариантов предобработки текста с нарастающей агрессивностью: `raw` (исходный, для BERT), `basic_clean` (lowercase + URL), `normalized` (+ пунктуация, числа), `no_stopwords` (+ стоп-слова), `stemmed` (+ стемминг). Каждый вариант сохраняется в `data/prepared/text_{name}.csv` |

## Этап 4 — Обучение/дообучение модели (1–5 баллов)

### Обязательная часть

| Критерий | Баллы | Статус | Реализация |
|----------|-------|--------|------------|
| Построение не менее 2 моделей (деревья/ансамбли + нейронные сети) | 1 | Сделано | `model/trainer.py` — DistilBERT (нейросеть) с двумя режимами: `freeze_encoder=True` (только классификатор) и `freeze_encoder=False` (полное дообучение). Бустинг не использовался ввиду бессмысленности для текстовой классификации без эмбеддингов, вместо этого реализована более сложная архитектура на трансформерах с множеством дополнительных функций |

### Дополнительные баллы

| Критерий | Баллы | Статус | Реализация |
|----------|-------|--------|------------|
| Реализация дообучения предыдущей модели (без обучения с нуля) | 1-2 | Сделано | `python run.py update --name baseline_frozen_encoder_trump --base-model baseline_frozen_encoder` — загружает веса базовой модели и дообучает на новых данных |
| Разработка нескольких моделей с различной устойчивостью к входным данным | 1-2 | Сделано | 4 модели: `baseline_frozen_encoder` (общая), `baseline_frozen_encoder_tuned2018` (адаптирована к 2018), `baseline_frozen_encoder_trump` (специализирована на trump-новостях), `baseline_unfrozen_encoder` (полностью переобученная) |

## Этап 5 — Валидация модели (2–10 баллов)

### Обязательная часть

| Критерий | Баллы | Статус | Реализация |
|----------|-------|--------|------------|
| Оценка качества модели/моделей (hold-out/CV/TimeSeriesCV) | 1-3 | Сделано | `model/trainer.py` — hold-out валидация (train/test split по времени), метрики: accuracy, precision, recall, F1, ROC-AUC. Команда: `python run.py validate --model baseline_frozen_encoder --files data/processed/test1.csv` |
| Разработка хранилища версий моделей и контроль качества | 1-2 | Сделано | `model/registry.py` — `models/` директория с версиями, каждая модель содержит: `model.safetensors`, `config.json`, `tokenizer.json`, `metrics.json`. Команда `python run.py summary` выводит все модели и их метрики |

### Дополнительные баллы

| Критерий | Баллы | Статус | Реализация |
|----------|-------|--------|------------|
| Интерпретация прогнозов (LIME, SHAP, визуализация) | 1-3 | Сделано | `model/interpretability.py` — `AttentionInterpreter.explain()` извлекает attention weights из DistilBERT, выделяет top-N важных слов, вычисляет attention на пунктуацию и стоп-слова. Команда: `python run.py explain --model baseline_frozen_encoder --text "..."` |
| Мониторинг и обработка ситуаций model drift | 1-2 | Сделано | `model/streaming.py` — `StreamEmulator` детектирует drift по содержанию "trump" в батче. При превышении `trump_threshold` автоматически переключается на `baseline_frozen_encoder_trump`. Команда: `python run.py stream --trump-threshold 0.3` |

## Этап 6 — Обслуживание модели (1–6 баллов)

### Обязательная часть

| Критерий | Баллы | Статус | Реализация |
|----------|-------|--------|------------|
| Выбор и упаковка (сериализация) финальной модели | 1-2 | Сделано | `model/predictor.py` — модели сериализованы в `safetensors` формате (HuggingFace), загрузка через `BertPredictor(model_name)`. Поддержка inference на CPU/GPU |

### Дополнительные баллы

| Критерий | Баллы | Статус | Реализация |
|----------|-------|--------|------------|
| Мониторинг производительности (времени применения/памяти) | 1-2 | Сделано | `model/streaming.py` — real-time статистика в `_stats()`: `total_requests`, `processed`, `avg_latency_ms`, `throughput_rps`, `queue_size`, `batch_count`. Отчёт каждые N секунд (`--report-interval`) |
| Обеспечение гибкого прогноза на основе данных (выбор модели при аномальных значениях) | 1-2 | Сделано | Автоматическое переключение модели при drift (trump-detection), обработка пустых запросов (`--empty-ratio`), graceful degradation при ошибках |

## Этап 7 — Управление программой (3–11 баллов)

### Обязательная часть

| Критерий | Баллы | Статус | Реализация |
|----------|-------|--------|------------|
| Создание скрипта управления конвейером и обработка запросов (Inference, Update, Summary) | 2 | Сделано | `run.py` — CLI через `click`: `inference`, `predict`, `train`, `update`, `validate`, `compare`, `explain`, `summary`, `stream` |
| Написание документации к программной реализации (README и requirements) | 1 | Сделано | `README.md` — описание моделей, команды запуска, примеры. `requirements.txt` — зависимости |

### Дополнительные баллы

| Критерий | Баллы | Статус | Реализация |
|----------|-------|--------|------------|
| Построение расширенного отчета (dashboard) о работе системы | 1-2 | Сделано | `python run.py summary` — выводит сводку по всем моделям: количество моделей, метрики (F1, accuracy) по каждому датасету, список доступных датасетов. Реализовано в `model/registry.py` → `ModelRegistry.summary()` |
| Создание конфигурационного файла параметров всех компонентов системы | 1-2 | Сделано | `model/config.yaml` — параметры: `base_model`, `max_length`, `batch_size`, `learning_rate`, `epochs`, `freeze_encoder`, `models_dir`, `soft_labeling` |
