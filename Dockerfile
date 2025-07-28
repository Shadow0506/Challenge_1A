# Use Python 3.9 slim image for AMD64 compatibility
FROM --platform=linux/amd64 python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY process_pdfs.py .
COPY pdf_parser.py .
COPY heading_extractor.py .
COPY utils.py .

# Copy the trained model
COPY student_final/ ./student_final/

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Set environment variables
ENV PYTHONPATH=/app
ENV TRANSFORMERS_OFFLINE=1

# Set memory limits
ENV PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Command to run the application
CMD ["python", "process_pdfs.py"]
