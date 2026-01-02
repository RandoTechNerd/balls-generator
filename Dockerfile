# Base Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install Python dependencies
# (We upgrade pip first to avoid wheel build issues)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy App Code
COPY . .

# Expose Streamlit Port
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run App
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]