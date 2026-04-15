FROM python:3.11-slim
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip tools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# --- CRITICAL: VERIFY MODEL EXISTENCE ---
# This fails the build if the model didn't copy, saving you debugging time.
RUN ls -la /app/models/fraud_model.pkl || (echo "ERROR: Model file not found in build context!" && exit 1)

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]