FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RAG_CORPUS_DIR=/app/corpus \
    RAG_BACKEND=none

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY corpus ./corpus
RUN pip install ".[serve]" && useradd --create-home --uid 10001 app

USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "grounded_rag.api:app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8000"]
