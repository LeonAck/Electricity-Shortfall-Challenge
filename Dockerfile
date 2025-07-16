# Use an official Python image as the base
FROM python:3.11-slim

# Set a working directory inside the container
WORKDIR /app

# Copy your requirements file and install dependencies
COPY requirements.txt .
COPY preprocessing.py .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Specify the default command to run when the container starts
CMD ["python", "main.py"]