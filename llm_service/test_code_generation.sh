#!/bin/bash
# Curl commands to test the Music AI Assistant LLM Service API

# Set the API base URL
API_BASE="http://localhost:8081"

# 1. Test the health check endpoint
echo "========== TESTING HEALTH CHECK ENDPOINT =========="
curl -s -X GET "$API_BASE/api/health"
echo -e "\n\n"

# 2. Test code generation with Gemini LLM (default)
echo "========== TESTING CODE GENERATION WITH GEMINI LLM =========="
curl -s -X POST "$API_BASE/api/generate-code" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the top 5 most popular artists in the dataset?"
  }' | jq .
echo -e "\n\n"

# 3. Test code generation with Gemma LLM (explicit)
echo "========== TESTING CODE GENERATION WITH GEMMA LLM =========="
curl -s -X POST "$API_BASE/api/generate-code" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Show me the distribution of songs by year with a bar chart",
    "llm_type": "gemma"
  }' | jq .
echo -e "\n\n"

# 4. Test with a visualization request
echo "========== TESTING VISUALIZATION REQUEST =========="
curl -s -X POST "$API_BASE/api/generate-code" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Create a scatter plot comparing song duration and popularity"
  }' | jq .
echo -e "\n\n"

# 5. Test with a complex analysis request
echo "========== TESTING COMPLEX ANALYSIS REQUEST =========="
curl -s -X POST "$API_BASE/api/generate-code" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Analyze the relationship between song release year and audio features like danceability and energy"
  }' | jq .
echo -e "\n\n"

# 6. Test with a simple data exploration request
echo "========== TESTING SIMPLE DATA EXPLORATION REQUEST =========="
curl -s -X POST "$API_BASE/api/generate-code" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What columns are available in the music dataset and what are their data types?"
  }' | jq .
echo -e "\n\n"

# 7. Test with an invalid or ambiguous request
echo "========== TESTING INVALID/AMBIGUOUS REQUEST =========="
curl -s -X POST "$API_BASE/api/generate-code" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Make the music better"
  }' | jq .
echo -e "\n\n"

# 8. Test alternate LLM with specific analysis
echo "========== TESTING SPECIFIC ANALYSIS WITH ALTERNATE LLM =========="
curl -s -X POST "$API_BASE/api/generate-code" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Find the correlation between song tempo and popularity",
    "llm_type": "gemma"
  }' | jq .
echo -e "\n\n"

echo "All tests completed."
