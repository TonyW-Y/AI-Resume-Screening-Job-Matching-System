FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Set environment variables
ENV TRANSFORMERS_OFFLINE=0
ENV PYTHONPATH=/app

# Expose port
EXPOSE 7860

# Run Gradio app
CMD ["python", "app/gradio_app.py"]