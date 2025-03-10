FROM ubuntu:22.04

# Set up timezone (needed for tzdata installation)
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3 \
    python3-pip \
    python3-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up Python
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install uv

# Set up working directory
WORKDIR /app

# Copy requirements
COPY requirements.in .
COPY requirements.txt .

# Install dependencies
RUN python3 -m pip install -r requirements.txt

# Copy the rest of the code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV LEROBOT_HOME=/root/.cache/lerobot
ENV PYTHONUNBUFFERED=1

# Default command (will be overridden by docker-compose)
CMD ["bash"] 