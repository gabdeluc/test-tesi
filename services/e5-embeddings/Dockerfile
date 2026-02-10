FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('agentlans/multilingual-e5-small-aligned-sentiment'); AutoModel.from_pretrained('agentlans/multilingual-e5-small-aligned-sentiment')"
COPY main.py .
EXPOSE 5002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5002"]