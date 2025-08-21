# Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install uv for dependency management
RUN pip install --no-cache-dir uv

# Copy everything needed for installation first
COPY pyproject.toml uv.lock* README.md ./
COPY src/ ./src/

# Install dependencies system-wide
RUN uv pip install --system --no-cache .

# Copy the Flask app entrypoint
COPY app.py ./app.py

# Set environment variables
ENV MODEL_VERSION="v1"
ENV PYTHONPATH="/app/src"

# Optional: expose Flask port
EXPOSE 5000

# Default command to run the app
CMD ["python", "app.py"]
