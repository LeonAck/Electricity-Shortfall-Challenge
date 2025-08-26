FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependencies and source
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY app.py ./app.py

# Install package in editable mode so imports work
RUN uv pip install --system --no-cache --editable .

# Optional: verify
RUN python -c "from electricity_forecast import __name__; print('Package available')"

# Set environment
ENV MODEL_VERSION="v1"

# Run the app
CMD ["python", "app.py"]

