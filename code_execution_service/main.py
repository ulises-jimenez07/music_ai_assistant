"""
Code Execution Service for the Music AI Assistant.
Handles safe execution of generated Python code.
"""

# Standard Library Imports
import base64
import logging
import os
import sys
import traceback
from contextlib import asynccontextmanager
from io import BytesIO
from typing import (  # Added Any for mypy fix
    Any,
    Optional,
)

# Third-Party Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI  # Removed unused HTTPException
from pydantic import BaseModel
from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safe_builtins
from RestrictedPython.PrintCollector import PrintCollector

# Local Application/Library Imports
# Add parent directory to path to import shared modules
# (Ideally, manage this with package structure or PYTHONPATH)
# pylint: disable=wrong-import-position
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import get_full_dataset
from llm_utils import validate_generated_code

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global variables
MUSIC_DF: Optional[pd.DataFrame] = None  # Added type hint


def initialize_data():
    """Initialize the music dataset."""
    # pylint: disable=global-statement
    global MUSIC_DF

    # Load the dataset
    MUSIC_DF = get_full_dataset()

    if MUSIC_DF is not None:
        # Pylint Fix: Use lazy formatting for logging
        logger.info("Initialized music dataset with %d rows", len(MUSIC_DF))
    else:
        logger.error("Failed to initialize music dataset.")


# Define lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(_app: FastAPI):  # Pylint Fix: Use underscore for unused 'app'
    """Load data on startup."""
    # Startup: Initialize data
    initialize_data()
    yield
    # Shutdown: Can add cleanup here if needed


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Music AI Assistant Code Execution Service",
    description="Service for safely executing generated Python code",
    lifespan=lifespan,
)

# Get environment variables
PORT = 8082


# Pydantic models for request/response
class CodeRequest(BaseModel):
    """Request model containing the Python code to execute."""  # Pylint Fix: Added docstring

    code: str


class ExecutionResponse(BaseModel):
    """Response model containing the execution results."""  # Pylint Fix: Added docstring

    success: bool
    output: str
    has_visualization: bool
    visualization: Optional[str] = None
    error: Optional[str] = None
    traceback: Optional[str] = None


def create_restricted_globals() -> dict[str, Any]:  # Added return type hint
    """Create restricted globals dictionary for safe code execution."""
    # Start with safe builtins
    safe_globals: dict[str, Any] = {
        "__builtins__": safe_builtins,
        "pd": pd,
        "plt": plt,
        "music_df": MUSIC_DF,
        # Pass the class itself; an instance will be created per execution
        "_print_": PrintCollector,
        "BytesIO": BytesIO,
        "base64": base64,
        "np": np,
        "sns": sns,
        "stats": scipy.stats,
    }

    return safe_globals


@app.post("/api/execute", response_model=ExecutionResponse)
async def execute_code(request: CodeRequest):
    """
    API endpoint to execute Python code safely.

    Request body:
        code: The Python code to execute

    Returns:
        JSON response with execution results (ExecutionResponse model)
    """
    try:
        code = request.code
        # Pylint Fix: Use lazy formatting for logging (no args needed here)
        logger.info("Received code execution request")

        # Validate the code first
        is_valid, reason = validate_generated_code(code)
        if not is_valid:
            logger.warning("Code validation failed: %s", reason)
            # Use the Pydantic model for consistency
            return ExecutionResponse(
                success=False,
                error=f"Code validation failed: {reason}",
                output="",
                has_visualization=False,
                visualization=None,
                traceback=None,
            )

        # Add code to capture the plot if generated
        code_with_capture = (
            code
            + """
\n# Capture visualization if any
viz_base64 = None # Initialize viz_base64
try:
    if plt.get_fignums(): # Check if any figures exist
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        viz_base64 = base64.b64encode(buf.read()).decode('utf-8')
finally:
    # Ensure all figures are closed regardless of success/failure
    plt.close('all')
"""
        )

        try:
            # Compile the code with restrictions
            byte_code = compile_restricted(code_with_capture, filename="<inline>", mode="exec")

            # Prepare the restricted globals and locals for this execution
            restricted_globals = create_restricted_globals()
            # mypy Fix: Added type hint here
            restricted_locals: dict[str, Any] = {}

            # Create an instance of PrintCollector for this execution
            _print_collector = restricted_globals["_print_"]()
            restricted_globals["_print_"] = _print_collector

            # Execute the code
            # Pylint Fix: Disable exec-used warning for this line
            # pylint: disable=exec-used
            exec(byte_code, restricted_globals, restricted_locals)

            # Get the printed output from the instance
            output = _print_collector.printed_text

            # Check if a visualization was generated
            visualization = restricted_locals.get("viz_base64")
            has_visualization = visualization is not None

            # Use the Pydantic model
            return ExecutionResponse(
                success=True,
                output=output if output else "",
                has_visualization=has_visualization,
                visualization=visualization,
                error=None,
                traceback=None,
            )
        except Exception as e:
            # Pylint Fix: Use lazy formatting and exc_info for traceback
            logger.error("Error executing restricted code: %s", e, exc_info=True)
            # Ensure plots are closed even if execution fails
            plt.close("all")
            # Use the Pydantic model
            return ExecutionResponse(
                success=False,
                error=str(e),
                traceback=traceback.format_exc(),
                output="",
                has_visualization=False,
                visualization=None,
            )
    except Exception as e:
        # Pylint Fix: Use lazy formatting and exc_info for traceback
        logger.error("Error processing request: %s", e, exc_info=True)
        # Ensure plots are closed on outer errors too
        plt.close("all")
        # Instead of raising HTTPException, return the standard error response
        # This keeps the API response structure consistent.
        return ExecutionResponse(
            success=False,
            error=f"Internal server error: {str(e)}",
            traceback=traceback.format_exc(),
            output="",
            has_visualization=False,
            visualization=None,
        )


@app.get("/api/health")
async def health_check():
    """
    API endpoint for health checks.

    Returns:
        JSON response with health status
    """
    data_loaded = MUSIC_DF is not None
    row_count = len(MUSIC_DF) if data_loaded and MUSIC_DF is not None else 0
    return {
        "status": "healthy",
        "service": "code-execution-service",
        "data_loaded": data_loaded,
        "row_count": row_count,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
