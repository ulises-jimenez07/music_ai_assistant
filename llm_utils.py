"""
A module for testing and interacting with various LLM models through LangChain.
Supports Gemini and Gemma models with fallback mechanisms.
"""

import json
import logging
import os
import time
from typing import (
    Any,
    Dict,
)

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

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

# System prompt template for code generation
SYSTEM_PROMPT = """
You are a helpful AI assistant specialized in analyzing music data. Your task is to:
1. Interpret natural language questions about music data
2. Generate Python code using pandas and matplotlib/seaborn to answer these questions
3. Ensure the code is safe, efficient, and follows best practices

The data is stored in a pandas DataFrame called 'music_df' which contains the following columns:
- From music_metadata: song_id, artist, title, release_year, genre, subgenre, lyrics_snippet, emotion, language, album, release_date, key, tempo, loudness, time_signature
- From music_characteristics: popularity, energy, danceability, valence, acousticness, instrumentalness, liveness, speechiness, duration_ms, explicit
- Additional music characteristics: good_for_party, good_for_work_study, good_for_relaxation, good_for_exercise, good_for_running, good_for_yoga, good_for_driving, good_for_social, good_for_morning
- Similar music recommendations: similar_artist_1, similar_song_1, similarity_score_1, similar_artist_2, similar_song_2, similarity_score_2, similar_artist_3, similar_song_3, similarity_score_3

Your generated code should:
- Be complete and executable
- Include appropriate visualizations when relevant
- Handle potential errors gracefully
- Be well-commented to explain your approach
- Return both the code and a natural language explanation of the results

DO NOT use any external libraries or APIs beyond pandas, matplotlib, seaborn, and numpy.
DO NOT attempt to access any files or resources beyond the provided DataFrame.
"""


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


def generate_code_for_question(question: str, schema: Dict[str, Any], llm_type=None) -> Dict[str, Any]:
    """
    Generate Python code to answer a natural language question about music data.
    Uses modern LangChain syntax with chain operators for more efficient composition.
    """
    try:
        # Get the LLM
        llm = get_llm(llm_type)

        # Define the output parser
        output_parser = JsonOutputParser()

        # Create prompt template
        prompt = PromptTemplate.from_template(
            """
{system_prompt}

Here's the schema of the music_df DataFrame:
```python
{schema}
```

User Question: {question}

Please generate Python code to answer this question. The code should:
1. Assume the data is already loaded in a pandas DataFrame called 'music_df'
2. Include appropriate visualizations if relevant
3. Return both the code and a natural language explanation of the results

Format your response as a JSON object with the following structure:
{{
  "code": "# Your complete Python code here",
  "explanation": "Your explanation of what the code does and the results",
  "visualization_type": "bar_chart|scatter_plot|histogram|pie_chart|line_chart|none",
  "requires_visualization": false
}}
"""
        )

        # Create the chain using pipe operators
        chain = (
            {
                "question": lambda x: x["question"],
                "system_prompt": lambda _: SYSTEM_PROMPT,
                "schema": lambda x: json.dumps(x["schema"], indent=2),
            }
            | prompt
            | llm
            | output_parser
        )

        # Run the chain
        try:
            result = chain.invoke({"question": question, "schema": schema})

            # Add LLM type to the result
            result["llm_used"] = llm_type or DEFAULT_LLM

            return result
        except Exception as e:
            # Handle parsing errors
            logger.warning("Failed to parse response: %s", str(e))

            # Attempt a more direct approach if the parsing fails
            response = llm.invoke(
                prompt.format(
                    system_prompt=SYSTEM_PROMPT,
                    schema=json.dumps(schema, indent=2),
                    question=question,
                )
            )

            # Try to extract JSON from response text
            response_text = response.content if hasattr(response, "content") else str(response)
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if 0 <= json_start < json_end:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
            else:
                # If no JSON found, create a structured response
                result = {
                    "code": response_text,
                    "explanation": "The model did not return a structured response. Please review the generated code.",
                    "visualization_type": "none",
                    "requires_visualization": False,
                }

            # Add LLM type to the result
            result["llm_used"] = llm_type or DEFAULT_LLM

            return result

    except Exception as e:
        logger.error("Error generating code: %s", str(e))
        return {
            "code": "# Error generating code",
            "explanation": f"An error occurred: {str(e)}",
            "visualization_type": "none",
            "requires_visualization": False,
            "error": str(e),
        }


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
                logger.error("Error invoking %s directly: %s", llm_type, str(e))
                return False

    except Exception as e:
        print(f"❌ Failed to initialize or use {llm_type}: {str(e)}")
        logger.error("Error testing %s: %s", llm_type, str(e))
        return False


if __name__ == "__main__":
    for llm_type in ["gemini", "gemma"]:
        print(f"\n===== TESTING {llm_type.upper()} =====")
        test_direct_llm_calls(llm_type)
