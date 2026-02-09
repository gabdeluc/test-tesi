FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e pyproject.toml

# Pre-download E5 model
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
    AutoTokenizer.from_pretrained('agentlans/multilingual-e5-small-aligned-sentiment'); \
    AutoModel.from_pretrained('agentlans/multilingual-e5-small-aligned-sentiment')"

COPY main.py .

EXPOSE 5002

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:5002/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5002"]