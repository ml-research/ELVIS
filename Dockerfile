# Use the official NVIDIA PyTorch image with CUDA support
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set the working directory inside the container
WORKDIR /app
RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime
# Install system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*


# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel
# Install Python dependencies with --no-cache-dir
WORKDIR /app

RUN pip install opencv-python==4.8.0.74
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt