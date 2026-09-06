FROM python:3.12-slim

WORKDIR /app

# Install system audio dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY dsp/ ./dsp/
COPY ml/ ./ml/
COPY experiments/ ./experiments/
COPY data/sample/ ./data/sample/

RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "ml.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
