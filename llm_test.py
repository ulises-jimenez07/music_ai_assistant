"""
A module for testing and interacting with various LLM models through LangChain.
Supports Gemini and Gemma models with fallback mechanisms.
"""

import logging
import os
import time

from dotenv import load_dotenv

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama.llms import OllamaLLM

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("llm_test")

# Load environment variables from .env file
load_dotenv()

# Constants
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.0-flash")
DEFAULT_LLM = os.environ.get("DEFAULT_LLM", "gemini")  # Options: "gemini" or "gemma"


def get_llm(llm_type=None):
    """
    Get the specified LLM model through LangChain.

    Args:
        llm_type: String indicating which LLM to use ('gemini' or 'gemma').
                 If None, uses the DEFAULT_LLM value.

    Returns:
        A configured LangChain LLM instance.

    Raises:
        Exception: If unable to initialize any LLM.
    """
    if llm_type is None:
        llm_type = DEFAULT_LLM

    try:
        if llm_type.lower() == "gemini":
            if os.environ.get("GOOGLE_API_KEY"):
                llm = ChatGoogleGenerativeAI(
                    model=GEMINI_MODEL_NAME,
                    google_api_key=os.environ.get("GOOGLE_API_KEY"),
                    temperature=0.2,
                )
                logger.info("Loaded Gemini model: %s", GEMINI_MODEL_NAME)
                return llm

            logger.warning("GOOGLE_API_KEY not set, falling back to Gemma")
            return get_llm("gemma")

        if llm_type.lower() == "gemma":
            try:
                llm = OllamaLLM(model="gemma3")
                logger.info("Loaded Gemma model from Ollama")
                return llm
            except ImportError:
                logger.error("Ollama not found. Please install Ollama and ensure it's running.")
                raise
            except Exception as e:
                logger.error("Error loading Gemma model from Ollama: %s", str(e))
                raise

        # If we get here, the LLM type is unknown
        logger.warning("Unknown LLM type: %s, falling back to %s", llm_type, DEFAULT_LLM)
        return get_llm(DEFAULT_LLM)

    except Exception as e:
        logger.error("Error loading LLM model: %s", str(e))
        raise


def test_direct_llm_calls(llm_type):
    """
    Test interaction with the LLM using LangChain's interface.
    """
    print(f"\n----- Testing direct LLM call with {llm_type.upper()} -----")
    try:
        # Get the LLM
        llm = get_llm(llm_type)
        print(f"✅ Successfully initialized {llm_type} LLM instance")

        # Test a simple prompt directly
        test_prompt = "Write a one-sentence description of what pandas is in Python."
        print(f"Sending test prompt to {llm_type}...")

        # The specific invocation depends on the LLM type, but we'll try to handle both
        start_time = time.time()

        try:
            # For most LangChain LLMs
            response = llm.invoke(test_prompt)
            print(f"✅ Successfully received response from {llm_type} in {time.time() - start_time:.2f} seconds")
            print(f"Response: {response}")
            return True
        except AttributeError:
            # Some older LLM interfaces might use different methods
            try:
                response = llm(test_prompt)
                print(f"✅ Successfully received response from {llm_type} in {time.time() - start_time:.2f} seconds")
                print(f"Response: {response}")
                return True
            except Exception as e:
                print(f"❌ Error invoking {llm_type} directly: {str(e)}")
                logger.error("Error invoking %s directly: %s", llm_type, str(e), exc_info=True)
                return False

    except Exception as e:
        print(f"❌ Failed to initialize or use {llm_type}: {str(e)}")
        logger.error("Error testing %s: %s", llm_type, str(e), exc_info=True)
        return False


if __name__ == "__main__":
    for llm_type in ["gemini", "gemma"]:
        print(f"\n===== TESTING {llm_type.upper()} =====")
        test_direct_llm_calls(llm_type)
