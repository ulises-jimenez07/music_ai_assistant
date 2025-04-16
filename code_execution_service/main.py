"""
Code Execution Service for the Music AI Assistant.
Handles safe execution of generated Python code.
"""

import base64
import logging
import os
import sys
import traceback
from contextlib import asynccontextmanager
from io import BytesIO
from typing import (
    Any,
    Optional,
)

import matplotlib.pyplot as plt

# Add potential imports that were conditionally imported inside functions
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safe_builtins
from RestrictedPython.PrintCollector import PrintCollector

# Add parent directory to path to import shared modules
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
MUSIC_DF = None


def initialize_data():
    """Initialize the music dataset."""
    # pylint: disable=global-statement
    global MUSIC_DF

    # Load the dataset
    MUSIC_DF = get_full_dataset()

    if MUSIC_DF is not None:
        logger.info("Initialized music dataset with %d rows", len(MUSIC_DF))
    else:
        logger.error("Failed to initialize music dataset.")


# Define lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Startup: Initialize data
    initialize_data()
    yield


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
    """Request model containing the Python code to execute."""

    code: str


class ExecutionResponse(BaseModel):
    """Response model containing the execution results."""

    success: bool
    output: str
    has_visualization: bool
    visualization: Optional[str] = None
    error: Optional[str] = None
    traceback: Optional[str] = None


def create_restricted_globals():
    """Create restricted globals for safe code execution."""
    # Start with safe builtins
    safe_globals = {
        "__builtins__": safe_builtins,
        "pd": pd,
        "plt": plt,
        "music_df": MUSIC_DF,
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
        JSON response with execution results
    """
    try:
        code = request.code
        logger.info("Received code execution request")

        # Validate the code first
        is_valid, reason = validate_generated_code(code)
        if not is_valid:
            return {
                "success": False,
                "error": reason,
                "output": "",
                "has_visualization": False,
                "visualization": None,
            }

        # Add code to capture the plot if generated
        code_with_capture = (
            code
            + """
# Capture visualization if any
if plt.get_fignums():
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    viz_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close('all')
else:
    viz_base64 = None
"""
        )

        try:
            # Compile the code with restrictions
            byte_code = compile_restricted(code_with_capture, filename="<inline>", mode="exec")

            # Prepare the restricted globals
            restricted_globals = create_restricted_globals()
            restricted_locals: dict[str, Any] = {}

            # Execute the code
            # pylint: disable=exec-used
            exec(byte_code, restricted_globals, restricted_locals)

            # Get the printed output
            print_collector = restricted_globals.get("_print_", lambda: "")
            # Ensure output is a string
            if callable(print_collector):
                output = print_collector()
                # If output is still not a string, convert it
                if not isinstance(output, str):
                    output = str(output)
            else:
                output = str(print_collector)

            # Check if a visualization was generated
            has_visualization = "viz_base64" in restricted_locals and restricted_locals["viz_base64"] is not None
            visualization = restricted_locals.get("viz_base64")

            return ExecutionResponse(
                success=True,
                output=output if output else "",
                has_visualization=has_visualization,
                visualization=visualization,
                error=None,
                traceback=None,
            )
        except Exception as e:
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
    return {
        "status": "healthy",
        "service": "code-execution-service",
        "data_loaded": MUSIC_DF is not None,
        "row_count": len(MUSIC_DF) if MUSIC_DF is not None else 0,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
