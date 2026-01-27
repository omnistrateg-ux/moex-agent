# MOEX Agent — Yandex Cloud Serverless Container
# Build: docker build -f deploy/Dockerfile -t moex-agent .
# Run:   docker run -p 8080:8080 --env-file .env moex-agent

FROM python:3.11-slim

# Metadata
LABEL maintainer="MOEX Agent Team"
LABEL description="AI-Powered Trading Signal Generator for Moscow Exchange"

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8080

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY moex_agent/ ./moex_agent/
COPY main.py .
COPY config.yaml .
COPY db/schema.sql ./db/

# Create data directories and initialize SQLite schema
RUN mkdir -p data logs && \
    sqlite3 data/moex_agent.sqlite < db/schema.sql

# Copy models
COPY models/ ./models/

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE ${PORT}

# Run application
CMD ["python", "main.py"]
