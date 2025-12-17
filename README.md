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

### 3. Настройка секретов

Создайте файл `.secret.env` в корне проекта:

```bash
cp .env .secret.env
```

Заполните реальными значениями для S3 и wandb.

### 4. Загрузка данных

```bash
uv run dvc pull
```

### 5. Shell completions (опционально)

Для fish:

```fish
uv run fashionctl --show-completion fish | source
```

Для постоянной установки:

```fish
uv run fashionctl --show-completion fish > ~/.config/fish/completions/fashionctl.fish
```

## Train

Запуск обучения:

```bash
uv run fashionctl models train --config train_visual
```

Запуск инференса:

```bash
uv run fashionctl models infer --checkpoint checkpoints/last.ckpt --sample-customers 10
```

### Результат инференса

Файл `predictions.csv` в формате:

```csv
customer_id,prediction
abc123def456,0108775015 0108775044 0111593001 ...
```

Колонка `prediction` содержит 12 рекомендованных `article_id`, разделённых пробелами.
