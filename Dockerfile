FROM python:3.10-slim

# Install system dependencies for ANTs
RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    git \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire app
COPY . .

# Expose the port Gradio runs on
EXPOSE 7860

# Run your Gradio app
CMD ["python", "app.py"]
