# Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install uv for dependency management
RUN pip install --no-cache-dir uv

# Copy everything needed for installation first
COPY pyproject.toml uv.lock* README.md ./
COPY src/ ./src/
COPY app.py ./app.py

# Clean old builds and force fresh install
RUN rm -rf dist *.egg-info __pycache__ && \
    uv pip install --system --no-cache . --verbose

RUN echo "=== Checking installed package ===" \
    && grep -R "scripts" /usr/local/lib/python3.11/site-packages || echo "No scripts found"

# Optional: verify
RUN python -c "from electricity_forecast import __name__; print('Package available')"


# Set environment variables
ENV MODEL_VERSION="v1"
ENV PYTHONPATH="/app/src"

# Default command to run the app
CMD ["python", "app.py"]
