FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-serve.txt

COPY config.yaml ./
COPY src ./src
COPY app ./app
COPY data/lexicons ./data/lexicons

ENV PYTHONPATH=/app
ENV LOKY_MAX_CPU_COUNT=4

EXPOSE 8000

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
