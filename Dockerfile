FROM python:3.9-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application files
COPY . .

# Expose ports for FastAPI (9000) and MLflow (5000)
EXPOSE 9000 5000

# Start MLflow first, then FastAPI
CMD mlflow server --host 0.0.0.0 --port 5000 & uvicorn main:app --host 0.0.0.0 --port 9000 --reload
