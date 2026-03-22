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

## HF Space

Демо: https://huggingface.co/spaces/Nofer/FakeNewsClassifier_mlops

## Обработка данных:
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
