# Fashion Recommender

Рекомендательная система для посетителей H&M с учётом визуальных эмбедингов товаров.

## Инструменты

- **Управление зависимостями:** [uv](https://docs.astral.sh/uv/)
- **Логирование экспериментов:** [wandb](https://wandb.ai/guzbkm-higher-school-of-economics/mlops-project)

## Setup

### 1. Установка uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Создание окружения и установка зависимостей

```bash
uv sync
```

### 3. Настройка Weights & Biases

Создайте файл `.secret.env` в корне проекта и добавьте ваш API ключ:

```bash
cp .env .secret.env
# Отредактируйте .secret.env и замените WANDB_API_KEY на ваш ключ
```

### 4. Загрузка данных

Скачайте датасет:

```bash
uv run fashionctl datasets download --all
```

Эта команда загрузит:
- `articles.csv` - метаданные товаров (~35 MB)
- `customers.csv` - данные о клиентах
- `transactions_train.csv` - история покупок
- `images/` - изображения товаров

Проверьте загруженные данные:

```bash
uv run fashionctl datasets list
```

Вы увидите статус каждого датасета (local/remote).

### 5. Shell completions (опционально)

Для fish:

```fish
uv run fashionctl --show-completion fish | source
```

Для постоянной установки:

```fish
uv run fashionctl --show-completion fish > ~/.config/fish/completions/fashionctl.fish
```

## 6. Обучение и инференс

### 6.1 Обучение модели

Запустите обучение модели с визуальными эмбеддингами:

```bash
uv run fashionctl models train --config train_visual
```

После завершения обучения чекпоинты сохраняются в директории `checkpoints/`:
- `last.ckpt` - последний чекпоинт
- `visual-epoch=XX-val_loss=Y.YY.ckpt` - лучшие чекпоинты по метрике

Процесс обучения логируется в [Weights & Biases](https://wandb.ai/guzbkm-higher-school-of-economics/mlops-project).

### 6.2 Инференс

Запустите инференс с обученной моделью:

```bash
uv run fashionctl models infer --checkpoint checkpoints/last.ckpt --sample-customers 10
```

Результат сохраняется в файл `predictions.csv` в корне проекта:

```csv
customer_id,prediction
abc123def456,0108775015 0108775044 0111593001 ...
```

Колонка `prediction` содержит 12 рекомендованных `article_id`, разделённых пробелами.
