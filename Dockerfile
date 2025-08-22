# Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install uv for dependency management
RUN pip install --no-cache-dir uv

# Copy everything needed for installation first
COPY pyproject.toml uv.lock* README.md ./
COPY src/ ./src/

# Clean old builds and force fresh install
RUN rm -rf dist *.egg-info __pycache__ && \
    uv pip install --system --no-cache .

RUN echo "=== Checking installed package ===" \
    && grep -R "scripts" /usr/local/lib/python3.11/site-packages || echo "No scripts found"

# Copy the Flask app entrypoint
COPY app.py ./app.py

# Set environment variables
ENV MODEL_VERSION="v1"
ENV PYTHONPATH="/app/src"

# Default command to run the app
CMD ["python", "app.py"]
