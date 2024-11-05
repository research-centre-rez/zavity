# Start from a base image
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

COPY /devernay/devernay /usr/local/bin/devernay
RUN chmod +x /usr/local/bin/devernay

# Set the entrypoint to allow arguments to be passed
ENTRYPOINT ["python", "src/scripts/main.py"]
