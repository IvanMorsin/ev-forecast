FROM python:3.11-slim

# системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# рабочая директория
WORKDIR /app

# копируем зависимости
COPY requirements.txt .

# ставим python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# копируем весь проект
COPY . .

# порт (Render сам прокинет)
EXPOSE 8501

# запуск streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]