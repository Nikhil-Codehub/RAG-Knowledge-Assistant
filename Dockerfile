# 1. Base Image (Lightweight Python)
FROM python:3.10-slim

# 2. Set Working Directory inside container
WORKDIR /app

# 3. Install System Dependencies (Optional but good for PDF/C++ tools)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy Requirements first (Caching layer optimization)
COPY requirements.txt .

# 5. Install Python Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the application code
COPY . .

# 7. Create directory for temp docs inside container
RUN mkdir -p temp_docs

# 8. Expose Streamlit Port
EXPOSE 8501

# 9. Healthcheck (Optional: Checks if app is running)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 10. Command to run the app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]