# Stage 1: Install dependencies
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir --prefix=/install .

# Stage 2: Download models at build time
FROM python:3.12-slim AS models

COPY --from=builder /install /usr/local

RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True); \
SentenceTransformer('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('Models downloaded')"

# Stage 3: Runtime image
FROM python:3.12-slim

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy cached models from model stage
COPY --from=models /root/.cache/huggingface /root/.cache/huggingface

# Copy application code
COPY src/ src/
COPY schemas/ schemas/

# Create non-root user
RUN groupadd -r rtfm && useradd -r -g rtfm -d /app rtfm \
    && mkdir -p /app/docs /tmp/rtfm_uploads \
    && mv /root/.cache /app/.cache \
    && chown -R rtfm:rtfm /app /tmp/rtfm_uploads

USER rtfm

# HuggingFace cache for the non-root user
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "rtfm.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
