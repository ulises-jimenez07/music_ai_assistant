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
    Tuple,
)

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama.llms import OllamaLLM

# data utils for the schema
from data_utils import get_dataset_schema

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("llm_utils")

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
                custom_endpoint = os.environ.get("OLLAMA_HOST")
                endpoint = "http://host.docker.internal:11434" if custom_endpoint else "http://localhost:11434"
                llm = OllamaLLM(model="gemma3", base_url=endpoint)
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
            logger.warning(e)
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


def validate_generated_code(code: str) -> Tuple[bool, str]:
    """
    Validate the generated code for security issues.

    Args:
        code: The generated Python code

    Returns:
        Tuple of (is_valid, reason)
    """
    # List of forbidden patterns
    forbidden_patterns = [
        "import os",
        "import sys",
        "import subprocess",
        "import shutil",
        "__import__",
        "eval(",
        "exec(",
        "open(",
        "file(",
        "os.system",
        "os.popen",
        "os.spawn",
        "subprocess.run",
        "subprocess.Popen",
        "!rm",
        "!del",
        "!copy",
        "!mv",
        "!cp",
        "!chmod",
        "!chown",
        "requests.get",
        "requests.post",
        "urllib",
        "http.client",
        "socket.",
        "ftplib",
        "smtplib",
    ]

    # Check for forbidden patterns
    for pattern in forbidden_patterns:
        if pattern in code:
            return False, f"Code contains forbidden pattern: {pattern}"

    # Check for imports other than allowed ones
    import_lines = [
        line.strip() for line in code.split("\n") if line.strip().startswith("import ") or " import " in line
    ]
    allowed_imports = [
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scipy",
        "math",
        "random",
        "datetime",
        "collections",
        "re",
    ]

    for line in import_lines:
        is_allowed = False
        for allowed in allowed_imports:
            if allowed in line:
                is_allowed = True
                break
        if not is_allowed and line:
            return False, f"Code contains forbidden import: {line}"

    return True, "Code passed security validation"


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


def get_example_questions():
    """Get example questions for the UI."""
    return [
        "What are the top 5 most popular songs in the dataset?",
        "Show me the distribution of energy levels across different genres",
        "Which artists have the highest average danceability scores?",
        "Is there a correlation between valence and energy in the dataset?",
        "What's the average tempo by decade?",
        "Which emotions are most common in songs with high popularity?",
        "What songs are good for exercise with high energy levels?",
        "Show me similar artists to the most popular artists in the dataset",
        "Which genres are best for work/study based on the data?",
        "Compare the characteristics of songs good for parties vs. relaxation",
    ]


def test_code_generation(llm_type=None, num_questions=3):
    """
    Test the code generation functionality using example questions and the dataset schema.

    Args:
        llm_type: String indicating which LLM to use ('gemini' or 'gemma').
                 If None, uses the DEFAULT_LLM value.
        num_questions: Number of questions to test (default: 3)

    Returns:
        A list of results from the code generation.
    """
    print(f"\n----- Testing code generation with {llm_type or DEFAULT_LLM} -----")

    try:
        # Get the schema
        schema = get_dataset_schema()
        print("✅ Successfully loaded dataset schema")

        # Get example questions
        questions = get_example_questions()
        num_questions = min(num_questions, len(questions))

        test_questions = questions[:num_questions]

        results = []

        # Test each question
        for i, question in enumerate(test_questions):
            print(f'\nTesting question {i+1}/{num_questions}: "{question}"')

            start_time = time.time()
            result = generate_code_for_question(question, schema, llm_type)
            elapsed_time = time.time() - start_time

            result["question"] = question
            result["elapsed_time"] = elapsed_time

            print(f"✅ Generated code in {elapsed_time:.2f} seconds")
            print(f"- Visualization type: {result.get('visualization_type', 'none')}")

            # Get first 3 lines of the code
            code_preview = "\n".join(result.get("code", "").split("\n")[:3]) + "..."
            print(f"- Code preview: \n{code_preview}")

            results.append(result)

        print(f"\n✅ Successfully tested {num_questions} questions")
        return results

    except Exception as e:
        logger.error("Error testing code generation: %s", str(e))
        print(f"❌ Error testing code generation: {str(e)}")
        return []


if __name__ == "__main__":
    for llm_type in ["gemini", "gemma"]:
        print(f"\n===== TESTING {llm_type.upper()} =====")
        test_direct_llm_calls(llm_type)
        test_code_generation(llm_type, num_questions=2)
