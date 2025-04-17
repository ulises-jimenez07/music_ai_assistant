# Streamlit App

The Streamlit App serves as the frontend for the Music AI Assistant, providing a user-friendly web interface for asking questions about music data and viewing the results.

## Overview

The Streamlit App is the central component that:

1. Provides a user interface for asking questions about music data
2. Coordinates communication between the LLM Service and Code Execution Service
3. Displays results, visualizations, and explanations to the user
4. Handles error states and service health monitoring

## User Interface Components

### Main Interface

- **Header**: Title and brief description of the application
- **Question Input**: Text input field for entering natural language questions
- **Ask Button**: Button to submit questions for processing
- **Results Display**: Area for displaying answers, visualizations, and explanations
- **Code and Output Expanders**: Collapsible sections showing generated code and raw output

### Sidebar

- **LLM Selection**: Radio buttons to choose between Gemini and Gemma
- **Health Check Button**: Button to manually check services health
- **Example Questions**: Clickable example questions for quick use

### System Status

- **Dataset Info**: Information about the loaded dataset (row count, etc.)
- **Sample Columns**: Overview of available data columns
- **Service Health**: Status indicators for each service
- **Last Check Time**: Timestamp of the last health check

## Key Functions

### `main()`

The main Streamlit application function that sets up the page and coordinates all components.

### `process_question(question, llm_type)`

Processes a user question by:
1. Sending the question to the LLM Service to generate code
2. Sending the generated code to the Code Execution Service
3. Displaying the results to the user

### `check_services_health()`

Checks if the required microservices are up and running by:
1. Sending health check requests to each service
2. Collecting status information
3. Determining overall system health

### `process_health_check()`

Handles service health checking logic and updates session state.

### `display_response(result, execution_result, llm_type)`

Displays the response from the LLM and code execution services, including:
1. Natural language explanation
2. Visualizations (if any)
3. Error messages (if execution failed)
4. Generated code and raw output in expanders

### `setup_sidebar()`

Sets up the sidebar with settings and example questions.

### `display_system_status(col)`

Displays system status information in the given column.

## Service Communication

The app communicates with backend services using HTTP requests:

### LLM Service

- **Endpoint**: `{LLM_SERVICE_URL}/api/generate-code`
- **Method**: POST
- **Payload**: Question and LLM type
- **Response**: Generated code, explanation, and metadata

### Code Execution Service

- **Endpoint**: `{CODE_EXECUTION_SERVICE_URL}/api/execute`
- **Method**: POST
- **Payload**: Generated code
- **Response**: Execution results, output, and visualizations

### Health Checks

- **Endpoints**:
  - `{LLM_SERVICE_URL}/api/health`
  - `{CODE_EXECUTION_SERVICE_URL}/api/health`
- **Method**: GET
- **Response**: Service status information

## Error Handling

The app includes comprehensive error handling for:

1. **Service Unavailability**: Displays appropriate messages when services are down
2. **Code Generation Errors**: Shows error messages from the LLM Service
3. **Code Execution Errors**: Displays execution errors and tracebacks
4. **Visualization Errors**: Handles errors in displaying visualizations

## Session State Management

The app uses Streamlit's session state to maintain state between interactions:

- `services_checked`: Whether services have been checked
- `services_healthy`: Whether all services are healthy
- `service_status`: Status of each service
- `dataset_info`: Information about the loaded dataset
- `run_query`: Whether to run a query on page load
- `last_health_check`: Timestamp of the last health check
- `force_health_check`: Whether to force a health check

## Configuration

The app is configured through environment variables:

- `LLM_SERVICE_URL`: URL of the LLM service (default: http://localhost:8081)
- `CODE_EXECUTION_SERVICE_URL`: URL of the code execution service (default: http://localhost:8082)
- `REQUEST_TIMEOUT`: Timeout for API requests in seconds (default: 30)

## Docker Configuration

The app runs in a Docker container with:
- Python 3.12 base image
- Streamlit running on port 8501
- Environment variables for service URLs

## Integration with Other Services

The Streamlit App integrates with:
- The LLM Service for generating code
- The Code Execution Service for executing code and getting results

It serves as the orchestrator that brings these services together to provide a seamless user experience.
