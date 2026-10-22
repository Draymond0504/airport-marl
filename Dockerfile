FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV APRONOPS_DB_PATH=/app/data/apronops.db

EXPOSE 8000 8501
