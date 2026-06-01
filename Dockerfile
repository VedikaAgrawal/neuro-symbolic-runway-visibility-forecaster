# Use an official lightweight Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables for compilation, threading, and optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    KMP_DUPLICATE_LIB_OK=True

# Set work directory
WORKDIR /app

# Install system build dependencies, keeping cache footprint clean
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 1. OPTIMIZATION: Install CPU-only PyTorch directly from PyTorch official index.
# This prevents downloading massive CUDA GPU libraries (reducing image size by 2GB+).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. OPTIMIZATION: Use pip to install other requirements directly.
# This respects our CPU-only PyTorch and skips installing massive GPU packages like triton/CUDA.
RUN pip install --no-cache-dir \
    "pandas>=2.0.0,<3.0.0" \
    "numpy>=1.24.0,<2.0.0" \
    "matplotlib>=3.7.0,<4.0.0" \
    "seaborn>=0.12.0,<0.14.0" \
    "pymongo>=4.0.0,<5.0.0" \
    "boto3>=1.26.0,<2.0.0" \
    "flask>=2.2.0,<4.0.0" \
    "scikit-learn>=1.2.0,<2.0.0" \
    "xgboost>=1.7.0,<4.0.0" \
    "z3-solver>=4.12.0.0,<5.0.0.0"

# Copy application source files
COPY src /app/src
COPY scripts /app/scripts

# Expose port 5050 for the Flask dashboard app
EXPOSE 5050

# Start Flask Web Server
CMD ["python", "-m", "src.app.main"]
