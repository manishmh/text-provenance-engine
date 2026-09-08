# --- Build stage ---
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build deps for psycopg2-binary
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/

# Non-editable install so the package is fully in site-packages
RUN pip install --no-cache-dir '.[api,api-pg]'

# --- Runtime stage ---
FROM python:3.10-slim AS runtime

WORKDIR /app

# Install runtime lib for psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Create non-root user
RUN groupadd -r provenance && useradd -r -g provenance provenance
RUN mkdir -p /app/data && chown -R provenance:provenance /app

USER provenance

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "--factory", "provenance.api.app:create_app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
