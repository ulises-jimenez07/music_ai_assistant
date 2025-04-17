# LLM Service

The LLM Service is a microservice responsible for generating Python code from natural language questions about music data. It serves as the "brain" of the Music AI Assistant, interpreting user questions and translating them into executable code.

## Overview

The LLM Service is implemented as a FastAPI application that:

1. Receives natural language questions from the Streamlit frontend
2. Uses LangChain with either Gemini or Gemma to generate Python code
3. Validates the generated code for security issues
4. Returns the code along with explanations and metadata

## API Endpoints

### POST `/api/generate-code`

Generates Python code for a natural language question about music data.

- **Request Body**:
  ```json
  {
    "question": "What are the top 5 most popular songs?",
    "llm_type": "gemini"  // Optional, defaults to environment variable
  }
  ```

- **Response**:
  ```json
  {
    "code": "# Python code to answer the question",
    "explanation": "Natural language explanation of what the code does",
    "visualization_type": "bar_chart",
    "requires_visualization": true,
    "validation": {
      "is_valid": true,
      "reason": "Code passed security validation"
    },
    "llm_used": "gemini"
  }
  ```

### GET `/api/health`

Health check endpoint to verify the service is running.

- **Response**:
  ```json
  {
    "status": "healthy",
    "service": "llm-service"
  }
  ```

## Data Models

### `QuestionRequest`

Pydantic model for code generation requests:
- `question`: The natural language question about music data
- `llm_type`: Optional string indicating which LLM to use ("gemini" or "gemma")

### `CodeResponse`

Pydantic model for generated code responses:
- `code`: The generated Python code
- `explanation`: Natural language explanation of what the code does
- `visualization_type`: Type of visualization (e.g., "bar_chart", "scatter_plot")
- `requires_visualization`: Boolean indicating if visualization is needed
- `validation`: Dictionary with validation results
- `llm_used`: Which LLM was used for generation

## Service Initialization

The service initializes on startup by:

1. Loading environment variables
2. Setting up logging
3. Initializing the dataset schema for code generation

The dataset schema is loaded once at startup and kept in memory for all subsequent requests, improving performance.

## Code Generation Process

When a question is received:

1. The service calls `generate_code_for_question()` from the `llm_utils` module
2. The LLM (either Gemini or Gemma) generates Python code based on the question
3. The generated code is validated using `validate_generated_code()`
4. The results are formatted as a `CodeResponse` and returned

## Configuration

The service is configured through environment variables:

- `LLM_SERVICE_PORT`: Port to run the service on (default: 8081)
- `DEFAULT_LLM`: Default LLM to use if not specified (default: "gemini")
- `GOOGLE_API_KEY`: API key for Google Gemini (if using Gemini)
- `OLLAMA_HOST`: Host for Ollama (if using Gemma)

## Docker Configuration

The service runs in a Docker container with:
- Python 3.12 base image
- Access to shared modules (data_utils.py, llm_utils.py)
- Access to the data directory
- Port 8081 exposed

## Error Handling

The service includes comprehensive error handling:
- Validation errors for generated code
- LLM connection and generation errors
- Request parsing errors

All errors are logged and appropriate HTTP status codes are returned.

## Integration with Other Services

The LLM Service is designed to work with:
- The Streamlit frontend, which sends questions and displays results
- Ollama (for Gemma) or Google Gemini API (for Gemini)

It does not directly interact with the Code Execution Service; that integration is handled by the Streamlit app.
