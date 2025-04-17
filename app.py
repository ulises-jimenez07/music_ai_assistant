"""
Main application for the Music AI Assistant.
Provides a Streamlit web interface for users to ask questions about music data using Gemini or Gemma (local).
"""

# pylint: disable=wrong-import-position, unused-import, too-many-locals, too-many-branches, too-many-statements
import base64
import io
import json
import logging
import os
from typing import (
    Any,
    Dict,
    Tuple,
)

import matplotlib

# plt is used for string pattern matching in code modification functions
import matplotlib.pyplot as plt  # noqa: F401

# Set the Matplotlib backend to 'Agg' to prevent opening windows
matplotlib.use("Agg")
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from requests.exceptions import ConnectionError as RequestConnectionError
from requests.exceptions import (
    RequestException,
    Timeout,
)

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Service URLs
LLM_SERVICE_URL = os.environ.get("LLM_SERVICE_URL", "http://localhost:8081")
CODE_EXECUTION_SERVICE_URL = os.environ.get("CODE_EXECUTION_SERVICE_URL", "http://localhost:8082")

# Timeout settings for API requests (in seconds)
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "30"))


class ServiceError(Exception):
    """Custom exception for service-related errors."""

    def __init__(self, message: str, status_code: int = None, service: str = None):
        self.message = message
        self.status_code = status_code
        self.service = service
        super().__init__(self.message)


def make_api_request(url: str, method: str = "get", json_data: Dict = None) -> Dict[str, Any]:
    """
    Make an API request with proper error handling.

    Args:
        url: The URL to make the request to
        method: The HTTP method to use (get or post)
        json_data: JSON data to send in the request body

    Returns:
        Dictionary containing the JSON response

    Raises:
        ServiceError: If the request fails
    """
    try:
        if method.lower() == "get":
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
        elif method.lower() == "post":
            response = requests.post(url, json=json_data, timeout=REQUEST_TIMEOUT)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        # Check if the request was successful
        response.raise_for_status()

        # Parse the JSON response
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON response from %s", url)
            raise ServiceError(f"Invalid JSON response from {url}", status_code=response.status_code) from exc

    except Timeout as exc:
        logger.error("Request to %s timed out after %s seconds", url, REQUEST_TIMEOUT)
        raise ServiceError(
            f"Service request timed out after {REQUEST_TIMEOUT} seconds", service=url.split("/")[2]
        ) from exc
    except RequestConnectionError as exc:
        logger.error("Connection error when connecting to %s", url)
        raise ServiceError("Could not connect to service", service=url.split("/")[2]) from exc
    except RequestException as e:
        status_code = e.response.status_code if hasattr(e, "response") else None
        logger.error("Request error for %s: %s", url, str(e))
        raise ServiceError(
            f"Service request error: {str(e)}", status_code=status_code, service=url.split("/")[2]
        ) from e


def get_dataset_info() -> Dict[str, Any]:
    """
    Get dataset information from the code execution service.

    Returns:
        Dictionary containing dataset information
    """
    try:
        data = make_api_request(f"{CODE_EXECUTION_SERVICE_URL}/api/health")

        return {
            "status": data.get("status", "unknown"),
            "row_count": data.get("row_count", 0),
            "data_loaded": data.get("data_loaded", False),
            "columns": data.get("columns", []),
        }
    except ServiceError as e:
        logger.error("Failed to get dataset info: %s", str(e))
        return {
            "status": "error",
            "row_count": 0,
            "data_loaded": False,
            "columns": [],
            "error": str(e),
        }


def generate_code_for_question(question: str, llm_type: str = None) -> Dict[str, Any]:
    """
    Generate Python code to answer a natural language question by calling the LLM service.

    Args:
        question: The natural language question about music data
        llm_type: The type of LLM to use ("gemini" or "gemma")

    Returns:
        Dictionary containing generated code, explanation, and metadata
    """
    try:
        # Prepare the request with additional instructions to avoid plt.show()
        # and use savefig approach instead to prevent separate windows
        modified_question = (
            question + " IMPORTANT: DO NOT use plt.show() in your code. Instead, save the figure "
            "to a variable so it can be displayed in the streamlit app."
        )

        payload = {"question": modified_question, "llm_type": llm_type}

        # Call the LLM service
        result = make_api_request(f"{LLM_SERVICE_URL}/api/generate-code", method="post", json_data=payload)

        # Validate the response
        required_fields = ["code", "explanation", "visualization_type", "requires_visualization"]
        missing_fields = [field for field in required_fields if field not in result]

        if missing_fields:
            logger.warning("LLM response missing fields: %s", ", ".join(missing_fields))
            # Add any missing fields with default values
            for field in missing_fields:
                if field == "code":
                    result[field] = "# Error: No code was generated"
                elif field == "explanation":
                    result[field] = "No explanation was provided by the LLM service."
                elif field in ["visualization_type", "requires_visualization"]:
                    result[field] = "none" if field == "visualization_type" else False

        # Fix the generated code to ensure plt.show() is not used
        code = result.get("code", "")
        if "plt.show()" in code:
            # Replace plt.show() with saving to BytesIO
            code = code.replace(
                "plt.show()",
                "# Save figure to buffer instead of showing\n"
                "import io, base64\n"
                "buf = io.BytesIO()\n"
                "plt.savefig(buf, format='png')\n"
                "buf.seek(0)\n"
                "# Convert to base64 for display in Streamlit\n"
                "img_str = base64.b64encode(buf.getvalue()).decode('utf-8')\n"
                "# Close the figure to prevent display in separate window\n"
                "plt.close()",
            )
            result["code"] = code

        return result

    except ServiceError as e:
        logger.error("LLM service error: %s", str(e))
        return {
            "code": "# Error generating code",
            "explanation": f"Failed to generate code: Service error - {str(e)}",
            "visualization_type": "none",
            "requires_visualization": False,
            "error": str(e),
            "llm_used": llm_type or "unknown",
        }


def execute_code_safely(code: str) -> Dict[str, Any]:
    """
    Execute the generated code by calling the code execution service.

    Args:
        code: The Python code to execute

    Returns:
        Dictionary containing execution results
    """
    try:
        # Modify the code to ensure any plt.show() calls are replaced with savefig
        # This prevents opening separate windows and captures the visualization
        modified_code = code
        if "plt.show()" in modified_code:
            # Replace plt.show() with code to save to BytesIO and encode as base64
            modified_code = modified_code.replace(
                "plt.show()",
                "import io, base64\n"
                "buf = io.BytesIO()\n"
                "plt.savefig(buf, format='png')\n"
                "buf.seek(0)\n"
                "plt.close()",  # Close the figure to prevent it from being displayed
            )
        elif "plt." in modified_code and ".show()" in modified_code:
            # Handle cases where plt might be aliased or show() is called on a specific figure
            modified_code += "\n\n# Close any open figures to prevent separate windows\nimport matplotlib.pyplot as plt\nplt.close('all')"

        # Prepare the request
        payload = {"code": modified_code}

        # Call the code execution service
        result = make_api_request(f"{CODE_EXECUTION_SERVICE_URL}/api/execute", method="post", json_data=payload)

        # Verify that required fields are present
        required_fields = ["success", "output", "has_visualization"]
        missing_fields = [field for field in required_fields if field not in result]

        if missing_fields:
            logger.warning("Code execution response missing fields: %s", ", ".join(missing_fields))
            # Add default values for missing fields
            for field in missing_fields:
                if field == "success":
                    result[field] = False
                elif field == "output":
                    result[field] = "No output was returned from the code execution service."
                elif field == "has_visualization":
                    result[field] = False

        # If visualization is claimed but not present, fix it
        if result.get("has_visualization", False) and not result.get("visualization"):
            result["has_visualization"] = False
            logger.warning("Code execution service claimed to have visualization but didn't provide it")

        return result

    except ServiceError as e:
        logger.error("Code execution service error: %s", str(e))
        return {
            "success": False,
            "error": f"Service error: {str(e)}",
            "output": "",
            "has_visualization": False,
            "visualization": None,
        }


def get_example_questions():
    """Get example questions for the UI."""
    return [
        "What are the top 5 most popular songs in the dataset?",
        "Which artists have the highest average danceability scores?",
        "Is there a correlation between valence and energy in the dataset?",
        "What emotions are most common in songs with high popularity?",
        "What songs are good for exercise with high energy levels?",
        "Which genres are best for work/study based on the data?",
        "Compare the characteristics of songs good for parties vs. relaxation",
    ]


def check_services_health() -> Tuple[bool, Dict[str, str]]:
    """
    Check if the required microservices are up and running.

    Returns:
        Tuple containing:
            - Boolean indicating if all services are healthy
            - Dictionary mapping service names to their health status
    """
    services_healthy = True
    service_status = {}

    # Check LLM service
    try:
        data = make_api_request(f"{LLM_SERVICE_URL}/api/health")
        service_status["llm_service"] = data.get("status", "unknown")
        if service_status["llm_service"] != "healthy":
            services_healthy = False
    except ServiceError as e:
        logger.error("Error checking LLM service health: %s", str(e))
        service_status["llm_service"] = "unreachable"
        services_healthy = False

    # Check code execution service
    try:
        data = make_api_request(f"{CODE_EXECUTION_SERVICE_URL}/api/health")
        service_status["code_execution_service"] = data.get("status", "unknown")
        if service_status["code_execution_service"] != "healthy":
            services_healthy = False

        # Also check if data is loaded
        if not data.get("data_loaded", False):
            service_status["data_status"] = "not_loaded"
            services_healthy = False
        else:
            service_status["data_status"] = "loaded"

    except ServiceError as e:
        logger.error("Error checking code execution service health: %s", str(e))
        service_status["code_execution_service"] = "unreachable"
        service_status["data_status"] = "unknown"
        services_healthy = False

    return services_healthy, service_status


def display_response(result, execution_result, llm_type):
    """
    Display the response from the LLM and code execution services.

    Args:
        result: Dictionary containing code generation results
        execution_result: Dictionary containing code execution results
        llm_type: The type of LLM used
    """
    # Display explanation
    st.markdown("### Answer")
    st.markdown(result.get("explanation", "No explanation provided."))

    # Move visualization to a dedicated expander instead of displaying directly
    if execution_result.get("has_visualization", False) and execution_result.get("visualization") is not None:
        with st.expander("View Visualization"):
            try:
                # Display the base64 encoded image within the expander
                viz_data = execution_result["visualization"]
                st.image(
                    io.BytesIO(base64.b64decode(viz_data)),
                    caption="Generated Visualization",
                    use_column_width=True,
                )
            except Exception as e:
                st.error(f"Error displaying visualization: {str(e)}")

    # Display error if execution failed
    if not execution_result.get("success", False):
        error_msg = execution_result.get("error", "Unknown error")
        st.error(f"Execution Error: {error_msg}")

    # Display code and output in expanders
    with st.expander("View Generated Code"):
        st.code(result.get("code", "# No code generated"), language="python")

    with st.expander("View Raw Output"):
        st.text(execution_result.get("output", "No output generated"))

    # Display which LLM was used
    st.info(f"Code generated using: {result.get('llm_used', llm_type).upper()}")


def process_health_check():
    """Handle service health checking logic."""
    services_healthy, service_status = check_services_health()
    st.session_state.services_healthy = services_healthy
    st.session_state.service_status = service_status
    st.session_state.last_health_check = pd.Timestamp.now()
    st.session_state.force_health_check = False  # Reset the force flag

    # Get dataset info if services are healthy
    if services_healthy:
        st.session_state.dataset_info = get_dataset_info()

    return services_healthy


def setup_sidebar():
    """Set up the sidebar with settings and example questions."""
    st.sidebar.title("Settings")
    llm_type = st.sidebar.radio(
        "Select LLM Engine:",
        ["gemini", "gemma"],
        index=0,
        help="Choose which Large Language Model to use for generating code",
    )

    # Option to check services health manually
    if st.sidebar.button("Check Services Health"):
        with st.spinner("Checking services health..."):
            services_healthy = process_health_check()
            if services_healthy:
                st.sidebar.success("Services health checked successfully!")
            else:
                st.sidebar.error("Some services are unhealthy!")

    # Example questions
    st.sidebar.title("Example Questions")
    example_questions = get_example_questions()

    for question in example_questions:
        if st.sidebar.button(question):
            st.session_state.question = question
            st.session_state.run_query = True
            # Update URL params
            st.query_params.question = question

    return llm_type


def display_system_status(col):
    """Display system status information in the given column."""
    with col:
        if st.session_state.services_healthy:
            st.markdown("### Dataset Info")
            st.markdown(f"**Rows:** {st.session_state.dataset_info.get('row_count', 0)}")

            # Display a few sample columns
            st.markdown("### Sample Columns")
            st.markdown("- **Metadata:** artist, title, genre, release_date, album, key")
            st.markdown("- **Audio Features:** popularity, energy, danceability, tempo, valence")
            st.markdown("- **Usage Context:** good_for_party, good_for_work_study, good_for_exercise")
            st.markdown("- **Similar Music:** similar_artist_1, similar_song_1, similarity_score_1")

        # Health status - display for both healthy and unhealthy states
        st.markdown("### System Status")
        for service, status in st.session_state.service_status.items():
            if status in ("healthy", "loaded"):
                st.markdown(f"✅ {service}: {status}")
            else:
                st.markdown(f"❌ {service}: {status}")

        # Last health check time
        if st.session_state.last_health_check:
            st.markdown(f"Last checked: {st.session_state.last_health_check.strftime('%H:%M:%S')}")

        # Show retry button when services are unhealthy
        if not st.session_state.services_healthy:
            if st.button("Retry Connection", key="bottom_retry"):
                st.session_state.force_health_check = True
                st.rerun()


def process_question(question, llm_type):
    """Process a user question and display results."""
    if not question:
        st.error("Please enter a question")
        return

    if not st.session_state.services_healthy:
        st.error("Cannot process question while services are unavailable")
        return

    with st.spinner(f"Analyzing your question using {llm_type.upper()} and generating insights..."):
        try:
            # Generate code for the question
            result = generate_code_for_question(question, llm_type)

            if "error" in result and result["error"]:
                st.error(f"Error generating code: {result['error']}")
            else:
                # Execute the generated code
                code = result.get("code", "")
                execution_result = execute_code_safely(code)

                # Display the results
                display_response(result, execution_result, llm_type)

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")


def main():
    """Main Streamlit application."""
    # Set page config
    st.set_page_config(
        page_title="Music AI Assistant",
        page_icon="🎵",
        layout="wide",
    )

    # Header
    st.title("🎵 Music AI Assistant")
    st.markdown("Ask questions about music data and get answers with visualizations")

    # Initialize session state variables
    if "services_checked" not in st.session_state:
        st.session_state.services_checked = False
        st.session_state.services_healthy = False
        st.session_state.service_status = {}
        st.session_state.dataset_info = {"status": "unknown", "row_count": 0, "columns": []}
        st.session_state.run_query = False
        st.session_state.last_health_check = None
        st.session_state.force_health_check = False

    # Create layout columns
    col1, col2 = st.columns([3, 1])

    # Add Retry Connection button at the top level
    with col2:
        retry_pressed = False
        if st.session_state.services_checked and not st.session_state.services_healthy:
            if st.button("Retry Connection", key="top_retry"):
                st.session_state.force_health_check = True
                retry_pressed = True
                st.session_state.services_checked = False  # Force a recheck

    # Check services health when needed
    current_time = pd.Timestamp.now()
    health_check_interval = pd.Timedelta(minutes=5)

    should_check_health = (
        not st.session_state.services_checked
        or st.session_state.force_health_check
        or st.session_state.last_health_check is None
        or current_time - st.session_state.last_health_check > health_check_interval
    )

    if should_check_health:
        with st.spinner("Checking services health..."):
            services_healthy = process_health_check()
            st.session_state.services_checked = True

            # Show confirmation if triggered by retry
            if retry_pressed:
                if services_healthy:
                    st.success("Connection successful! Services are now healthy.")
                else:
                    st.error("Services are still unavailable. Please check your configuration.")

    # Set up sidebar and get LLM type
    llm_type = setup_sidebar()

    # Main content
    with col1:
        # Question input
        question = st.text_input(
            "Your Question:",
            key="question",
            placeholder="e.g., What are the top 5 most popular songs?",
        )

        submit = st.button("Ask Question")

        # Show warning if services are not healthy
        if not st.session_state.services_healthy:
            st.error("⚠️ Some services are not available. Please check the System Status panel.")
            for service, status in st.session_state.service_status.items():
                if status not in ("healthy", "loaded"):
                    st.warning(f"{service}: {status}")

        # Process question when submitted
        if submit or ("run_query" in st.session_state and st.session_state.run_query):
            if "run_query" in st.session_state:
                st.session_state.run_query = False
            process_question(question, llm_type)

    # Display system status in the right column
    display_system_status(col2)


if __name__ == "__main__":
    main()
