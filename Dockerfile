# Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for Tesseract OCR and PDF processing
# This is the Linux equivalent of the installations we discussed for Windows
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Create directories for the application's input and output
RUN mkdir -p /app/input_files /app/organised_files /app/reports

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define the command to run the application
CMD ["python", "main.py"]
