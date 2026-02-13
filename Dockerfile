# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY model.py .
COPY api.py .

# Copy models
COPY models/voyagerModel.pth models/
COPY models/shakespeareModel.pth models/

# Copy training data (needed for vocabulary extraction)
COPY training/voyager_dense.txt training/
COPY training/shakespeare.txt training/

# Expose port (Cloud Run will set PORT env variable)
EXPOSE 8080

# Set environment variable for production
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "api.py"]

