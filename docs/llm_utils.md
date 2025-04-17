# LLM Utilities

The `llm_utils.py` module provides functionality for interacting with Large Language Models (LLMs) to generate Python code based on natural language questions about music data.

## Overview

This module serves as the bridge between user questions and executable Python code. It handles:

1. LLM initialization and management
2. Prompt engineering for code generation
3. Response parsing and validation
4. Fallback mechanisms between different LLM providers

## Supported LLM Models

The module supports two primary LLM options:

1. **Google Gemini** (cloud-based)
   - Requires a Google API key
   - Uses the `gemini-2.0-flash` model by default
   - Accessed through LangChain's `ChatGoogleGenerativeAI` interface

2. **Gemma** (local, via Ollama)
   - Runs locally through Ollama
   - Uses the `gemma3` model
   - Accessed through LangChain's `OllamaLLM` interface
   - Falls back to this option if Gemini is unavailable

## Key Functions

### `get_llm(llm_type=None)`

Initializes and returns an LLM instance based on the specified type.

- **Parameters**:
  - `llm_type`: String indicating which LLM to use ('gemini' or 'gemma'). If None, uses the default LLM.
- **Returns**: A configured LangChain LLM instance.
- **Behavior**:
  - Attempts to initialize the specified LLM
  - Implements fallback logic if the primary LLM is unavailable
  - Handles configuration through environment variables

### `generate_code_for_question(question, schema, llm_type=None)`

Generates Python code to answer a natural language question about music data.

- **Parameters**:
  - `question`: The natural language question
  - `schema`: Dictionary containing the dataset schema
  - `llm_type`: Optional string indicating which LLM to use
- **Returns**: Dictionary containing:
  - `code`: The generated Python code
  - `explanation`: Natural language explanation of what the code does
  - `visualization_type`: Type of visualization (if any)
  - `requires_visualization`: Boolean indicating if visualization is needed
  - `llm_used`: Which LLM was used for generation
- **Behavior**:
  - Uses LangChain's modern pipe operator syntax for chain composition
  - Includes error handling and fallback parsing for malformed responses
  - Structures the output as a consistent JSON object

### `validate_generated_code(code)`

Validates generated code for security issues.

- **Parameters**:
  - `code`: The generated Python code
- **Returns**: Tuple of (is_valid, reason)
- **Behavior**:
  - Checks for forbidden patterns like system imports, file operations, etc.
  - Validates imports against an allowlist
  - Returns detailed reason if validation fails

## Prompt Engineering

The module uses a carefully crafted system prompt that:

1. Defines the AI assistant's role as a music data analyst
2. Provides context about the available data and its structure
3. Sets expectations for code quality and safety
4. Specifies output format requirements

The prompt template combines:
- The system prompt
- The dataset schema
- The user's question
- Specific instructions for code generation

## Testing Functions

The module includes functions for testing LLM functionality:

- `test_direct_llm_calls`: Tests basic LLM interaction
- `test_code_generation`: Tests the full code generation pipeline with example questions

These functions are useful for verifying that the LLM services are working correctly during development and troubleshooting.
