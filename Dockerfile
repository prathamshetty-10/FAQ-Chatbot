FROM python:3.11-slim

WORKDIR /app

# Minimal system deps ONLY
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY . .

ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache
ENV PYTHONUNBUFFERED=1
ENV OPENROUTER_API_KEY=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
ENV OPENROUTER_MODEL=meta-llama/llama-3.2-3b-instruct:free

EXPOSE 3030

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "3030"]
