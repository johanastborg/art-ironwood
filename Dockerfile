FROM python:3.11-slim

# Install system dependencies for C++ compilation
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the local package (compiles C++ extension)
RUN pip install .

# Expose port (Cloud Run expects 8080)
EXPOSE 8080

# Run uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
