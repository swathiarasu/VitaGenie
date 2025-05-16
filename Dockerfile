# Dockerfile
FROM python:3.12.6-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY . .



# Make entrypoint scripts executable
RUN chmod +x entrypoints/*.sh

# Expose Streamlit's default port
EXPOSE 8501

# Default command: run the app
CMD ["bash", "entrypoints/run_app.sh"]
