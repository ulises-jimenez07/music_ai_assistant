# LLM Service Dockerfile
FROM python:3.12-slim
WORKDIR /app
# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copy shared modules and data
COPY .env .
COPY data_utils.py .
COPY llm_utils.py .
COPY ./data/ ./data/
# Copy service-specific code
COPY llm_service/main.py .
# Set environment variables
ENV PORT=8081
ENV DEBUG=False
# Set default Ollama host (can be overridden at runtime)
ENV OLLAMA_HOST=http://host.docker.internal:11434
# Expose the port
EXPOSE 8081
# Run the application
CMD ["python", "main.py"]
