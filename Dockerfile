# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=2.0.0 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    KMP_DUPLICATE_LIB_OK=True

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install "poetry==$POETRY_VERSION"

# Copy dependency specifications
COPY pyproject.toml poetry.lock /app/

# Install python packages
RUN poetry install --no-root --no-interaction --no-ansi --only main

# Copy application files
COPY src /app/src
COPY scripts /app/scripts
COPY data /app/data
COPY models /app/models

# Expose port 5000
EXPOSE 5000

# Start Flask Web Server
CMD ["python", "-m", "src.app.main"]
