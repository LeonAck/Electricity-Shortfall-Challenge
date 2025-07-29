# Use an official Python image as the base
FROM python:3.11-slim

# Set a working directory inside the container
WORKDIR /app

# Copy your requirements file and install dependencies
COPY requirements.txt .

# Copy the project files
COPY scripts/ ./scripts/

RUN pip install --no-cache-dir -r requirements.txt

# Add to your existing Dockerfile AFTER installing dependencies
COPY app.py /app/
RUN pip install flask google-cloud-storage

# Set model version as ENV (will be overridden at deploy time)
ENV MODEL_VERSION="v1"


COPY . .

# Specify the default command to run when the container starts
CMD ["python", "app.py"]