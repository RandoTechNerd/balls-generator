# Base Image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (GL required for some trimesh ops)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy App Code
COPY . .

# Expose Streamlit Port
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run App
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
