"""
LLM Service for the Music AI Assistant.
Handles question interpretation and code generation using LangChain with Gemini or Gemma LLMs.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import (
    Any,
    Dict,
    Optional,
)

# Add parent directory to path to import shared modules
# pylint: disable=wrong-import-position
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    HTTPException,
)
from pydantic import BaseModel

# Import local modules after modifying path
from data_utils import get_dataset_schema
from llm_utils import (
    generate_code_for_question,
    validate_generated_code,
)

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Get environment variables
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
PORT = int(os.environ.get("PORT", 8081))

# Global variables
DATASET_SCHEMA = None


# Define lifespan context manager
@asynccontextmanager
async def lifespan(_: FastAPI):
    """Lifespan event handler for startup and shutdown events."""
    # Initialize the data on startup
    initialize_data()
    yield
    # Cleanup code would go here if needed


# Initialize FastAPI app
app = FastAPI(
    title="Music AI Assistant LLM Service",
    description="Service for generating code to answer questions about music data",
    version="1.0.0",
    lifespan=lifespan,
)


class QuestionRequest(BaseModel):
    """Request model for code generation from natural language questions."""

    question: str
    llm_type: Optional[str] = None


class CodeResponse(BaseModel):
    """Response model for generated code and metadata."""

    code: str
    explanation: str
    visualization_type: str
    requires_visualization: bool
    validation: Dict[str, Any]
    llm_used: str


def initialize_data():
    """Initialize the dataset schema."""
    # Use function-level variable assignment instead of global statement
    # This still modifies the module-level variable
    # pylint: disable=global-statement
    global DATASET_SCHEMA
    # Get the schema for code generation
    DATASET_SCHEMA = get_dataset_schema()
    logger.info("Initialized dataset schema for code generation")


@app.post("/api/generate-code", response_model=CodeResponse)
async def generate_code(request: QuestionRequest):
    """
    API endpoint to generate code for a natural language question.
    Request body:
        question: The natural language question about music data
        llm_type: (optional) The type of LLM to use ("gemini" or "gemma")
    Returns:
        JSON response with generated code and explanation
    """
    try:
        question = request.question
        llm_type = request.llm_type
        logger.info("Received question: %s (LLM: %s)", question, llm_type or "default")

        # Generate code for the question
        result = generate_code_for_question(question, DATASET_SCHEMA, llm_type)

        # Validate the generated code
        code = result.get("code", "")
        is_valid, reason = validate_generated_code(code)

        # Add validation result to the response
        result["validation"] = {"is_valid": is_valid, "reason": reason}
        return result
    except Exception as e:
        logger.error("Error generating code: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/health")
async def health_check():
    """
    API endpoint for health checks.
    Returns:
        JSON response with health status
    """
    return {"status": "healthy", "service": "llm-service"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
