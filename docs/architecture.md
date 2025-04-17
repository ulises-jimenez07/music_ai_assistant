# Architecture Overview

This document provides an overview of the Music AI Assistant architecture, explaining how the different components interact with each other.

## System Architecture

```mermaid
graph TD
    User[User] -->|Asks questions| StreamlitApp[Streamlit App]
    StreamlitApp -->|Sends question| LLMService[LLM Service]
    LLMService -->|Generates code| StreamlitApp
    StreamlitApp -->|Sends generated code| CodeExecutionService[Code Execution Service]
    CodeExecutionService -->|Returns results & visualizations| StreamlitApp

    LLMService -->|Uses| Ollama[Ollama LLM]
    LLMService -->|Or uses| Gemini[Google Gemini API]

    CodeExecutionService -->|Loads & queries| MusicData[(Music Dataset)]

    subgraph "Frontend"
        StreamlitApp
    end

    subgraph "Backend Services"
        LLMService
        CodeExecutionService
    end

    subgraph "External Dependencies"
        Ollama
        Gemini
    end

    subgraph "Data Layer"
        MusicData
    end

    style StreamlitApp fill:#4CAF50,stroke:#388E3C,color:white
    style LLMService fill:#2196F3,stroke:#1976D2,color:white
    style CodeExecutionService fill:#FF9800,stroke:#F57C00,color:white
    style MusicData fill:#9C27B0,stroke:#7B1FA2,color:white
    style Ollama fill:#607D8B,stroke:#455A64,color:white
    style Gemini fill:#E91E63,stroke:#C2185B,color:white
```

## Component Interaction Flow

1. **User Interaction**:
   - User submits a natural language question about music data through the Streamlit web interface

2. **Code Generation**:
   - Streamlit app sends the question to the LLM Service
   - LLM Service uses either Gemini or Gemma (via Ollama) to generate Python code that answers the question
   - Generated code is returned to the Streamlit app

3. **Code Execution**:
   - Streamlit app sends the generated code to the Code Execution Service
   - Code Execution Service safely executes the code in a restricted environment
   - Code accesses the pre-loaded music dataset to perform analysis
   - Results and visualizations are returned to the Streamlit app

4. **Result Presentation**:
   - Streamlit app displays the results, visualizations, and explanations to the user

## Communication Protocols

- All inter-service communication uses HTTP/JSON
- The Streamlit app connects to services using environment variables:
  - `LLM_SERVICE_URL` (default: http://localhost:8081)
  - `CODE_EXECUTION_SERVICE_URL` (default: http://localhost:8082)

## Containerization

Each component runs in its own Docker container:
- `streamlit-app`: The frontend Streamlit application
- `llm-service`: The code generation service
- `code-execution-service`: The service that safely executes generated code

These containers are orchestrated using Docker Compose, with a shared network allowing them to communicate with each other.
