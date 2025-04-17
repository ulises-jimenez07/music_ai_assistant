# Music AI Assistant

A Streamlit-based application that allows users to ask questions about music data and receive answers with visualizations. The application uses AI to generate Python code based on natural language questions and executes the code safely to provide insights.

## Architecture


The application consists of three main components:

1. **Streamlit Frontend** - A web interface where users can ask questions about music data
2. **LLM Service** - Generates Python code based on natural language questions using either Gemini or Gemma
3. **Code Execution Service** - Safely executes the generated code and returns results with visualizations

For a detailed architecture overview with diagrams, see the [Architecture Documentation](docs/architecture.md).

## Documentation

Detailed documentation for each component:

- [Architecture Overview](docs/architecture.md) - System architecture and component interactions
- [Streamlit App](docs/streamlit_app.md) - Frontend web interface
- [LLM Service](docs/llm_service.md) - Code generation service
- [Code Execution Service](docs/code_execution_service.md) - Safe code execution service
- [LLM Utilities](docs/llm_utils.md) - Utilities for working with LLMs
- [Data Utilities](docs/data_utils.md) - Utilities for data loading and processing

## Prerequisites

- Docker and Docker Compose
- Ollama running locally (for the LLM service to connect to)

## Getting Started with Docker Compose

The project includes a Docker Compose configuration that makes it easy to run all services together.

### Using the Management Script

A management script is provided to simplify working with the Docker Compose setup:

```bash
# Make the script executable (if not already)
chmod +x manage-services.sh

# Start all services
./manage-services.sh start

# Check the status of services
./manage-services.sh status

# View logs from all services
./manage-services.sh logs

# View logs from a specific service
./manage-services.sh logs streamlit-app
./manage-services.sh logs llm-service
./manage-services.sh logs code-execution-service

# Stop all services
./manage-services.sh stop

# Restart all services
./manage-services.sh restart

# Rebuild services (after code changes)
./manage-services.sh build

# Stop and remove all containers and networks
./manage-services.sh down

# Show help
./manage-services.sh help
```

### Manual Docker Compose Commands

If you prefer to use Docker Compose directly:

```bash
# Start all services in the background
docker compose up -d

# View logs
docker compose logs -f

# Stop all services
docker compose stop

# Stop and remove containers, networks
docker compose down

# Rebuild and start services
docker compose up -d --build
```

## Accessing the Application

Once the services are running:

- Streamlit Frontend: http://localhost:8501
- LLM Service API: http://localhost:8081
- Code Execution Service API: http://localhost:8082

## Environment Variables

The Docker Compose configuration sets the following environment variables:

### Streamlit App
- `LLM_SERVICE_URL`: URL of the LLM service (default: http://llm-service:8081)
- `CODE_EXECUTION_SERVICE_URL`: URL of the code execution service (default: http://code-execution-service:8082)

### LLM Service
- `PORT`: Port to run the service on (default: 8081)
- `DEBUG`: Enable debug mode (default: False)
- `OLLAMA_HOST`: URL of the Ollama service (default: http://host.docker.internal:11434)

### Code Execution Service
- `PORT`: Port to run the service on (default: 8082)
- `DEBUG`: Enable debug mode (default: False)

## Development

If you need to modify the services:

1. Make your changes to the code
2. Rebuild the services: `./manage-services.sh build`
3. Restart the services: `./manage-services.sh restart`

## Troubleshooting

If you encounter issues:

1. Check the logs: `./manage-services.sh logs`
2. Ensure Ollama is running locally
3. Try rebuilding the services: `./manage-services.sh build`
4. Restart the services: `./manage-services.sh restart`

## Note on Ollama

The LLM service requires Ollama to be running locally. The Docker Compose configuration uses `host.docker.internal` to allow the container to access Ollama running on the host machine. This works on most Docker Desktop installations, but may require additional configuration on some Linux systems.
