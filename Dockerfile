# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code
COPY flask_app/ ./flask_app/

# Expose the port used by Flask
EXPOSE 5000

# Default command to run the app
CMD ["python", "flask_app/app.py"]