# Базовый образ Python
FROM python:3.11-slim

# Рабочая директория
WORKDIR /app

# Системные пакеты для сборки и LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Копируем зависимости Python и устанавливаем их
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Копируем код проекта
COPY ./model /app/model
COPY main.py /app/

# Volumes для данных и обученных моделей
VOLUME /app/data
VOLUME /app/model_output

# Порт FastAPI
EXPOSE 8000

# Запуск FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
