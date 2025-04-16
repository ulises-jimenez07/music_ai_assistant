#!/bin/bash
# Curl commands to test the Music AI Assistant Code Execution Service API

# Set the API base URL
API_BASE="http://localhost:8082"

# 1. Test the health check endpoint
echo "========== TESTING HEALTH CHECK ENDPOINT =========="
curl -s -X GET "$API_BASE/api/health"
echo -e "\n\n"

# 2. Test with basic print statement
echo "========== TESTING BASIC PRINT STATEMENT =========="
curl -s -X POST "$API_BASE/api/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "print(\"Hello from the Music AI Assistant!\")"
  }'

echo -e "\n\n"

# 3. Test with data exploration
echo "========== TESTING DATA EXPLORATION =========="
curl -s -X POST "$API_BASE/api/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "print(\"Data shape:\", music_df.shape)\nprint(\"Columns:\", music_df.columns.tolist())\nprint(\"First 2 rows:\")\nprint(music_df.head(2))"
  }'

echo -e "\n\n"

# 4. Test with data visualization (bar chart)
echo "========== TESTING DATA VISUALIZATION (BAR CHART) =========="
curl -s -X POST "$API_BASE/api/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "# Create a simple bar chart\ntry:\n    if \"year\" in music_df.columns:\n        top_years = music_df[\"year\"].value_counts().sort_values(ascending=False).head(5)\n        plt.figure(figsize=(10, 6))\n        top_years.plot(kind=\"bar\")\n        plt.title(\"Top 5 Years by Number of Songs\")\n        plt.xlabel(\"Year\")\n        plt.ylabel(\"Number of Songs\")\n        plt.tight_layout()\n        print(\"Bar chart created successfully\")\n    else:\n        print(\"Year column not found in dataset\")\n        # Create sample data for testing\n        plt.figure(figsize=(8, 5))\n        plt.bar([\"Sample 1\", \"Sample 2\", \"Sample 3\"], [5, 7, 3])\n        plt.title(\"Sample Bar Chart\")\n        plt.tight_layout()\nexcept Exception as e:\n    print(f\"Error creating visualization: {e}\")"
  }' | jq -r ".visualization"  | grep -v "null" | base64 -d > visualization.png

echo -e "\n\n"

# 5. Test with data visualization (scatter plot)
echo "========== TESTING DATA VISUALIZATION (SCATTER PLOT) =========="
curl -s -X POST "$API_BASE/api/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "# Create a scatter plot\ntry:\n    # Check if we have appropriate columns for a scatter plot\n    numeric_columns = music_df.select_dtypes(include=[\"number\"]).columns.tolist()\n    \n    if len(numeric_columns) >= 2:\n        # Use the first two numeric columns\n        x_col = numeric_columns[0]\n        y_col = numeric_columns[1]\n        \n        # Sample data to avoid overcrowding\n        sample_df = music_df.sample(min(50, len(music_df)))\n        \n        plt.figure(figsize=(10, 6))\n        plt.scatter(sample_df[x_col], sample_df[y_col], alpha=0.6)\n        plt.title(f\"{x_col} vs {y_col}\")\n        plt.xlabel(x_col)\n        plt.ylabel(y_col)\n        plt.tight_layout()\n        \n        print(f\"Created scatter plot of {x_col} vs {y_col}\")\n    else:\n        # Create sample data for testing\n        plt.figure(figsize=(8, 5))\n        plt.scatter([1, 2, 3, 4, 5], [5, 7, 2, 8, 4])\n        plt.title(\"Sample Scatter Plot\")\n        plt.xlabel(\"X\")\n        plt.ylabel(\"Y\")\n        plt.tight_layout()\n        \n        print(\"Created sample scatter plot (insufficient numeric columns)\")\nexcept Exception as e:\n    print(f\"Error creating visualization: {e}\")"
  }'

echo -e "\n\n"

# 6. Test with invalid code (syntax error)
echo "========== TESTING INVALID CODE (SYNTAX ERROR) =========="
curl -s -X POST "$API_BASE/api/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "# This has a syntax error\nprint(\"Starting\")\nif True\n    print(\"This will cause an error\")"
  }'

echo -e "\n\n"

# 7. Test with potentially unsafe code
echo "========== TESTING POTENTIALLY UNSAFE CODE =========="
curl -s -X POST "$API_BASE/api/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "# This attempts to import and use os module\nimport os\nos.system(\"echo Trying to execute a command\")"
  }'

echo -e "\n\n"

# 8. Test with complex analysis
echo "========== TESTING COMPLEX ANALYSIS =========="
curl -s -X POST "$API_BASE/api/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "# Perform a more complex analysis\ntry:\n    # Basic statistics\n    print(\"Dataset info:\")\n    print(f\"- Number of records: {len(music_df)}\")\n    print(f\"- Number of columns: {len(music_df.columns)}\")\n    \n    # Create a histogram of a numeric column if available\n    numeric_cols = music_df.select_dtypes(include=[\"number\"]).columns\n    \n    if len(numeric_cols) > 0:\n        # Take the first numeric column\n        num_col = numeric_cols[0]\n        \n        # Basic stats\n        print(f\"\\nStatistics for {num_col}:\")\n        print(music_df[num_col].describe())\n        \n        # Create histogram\n        plt.figure(figsize=(10, 6))\n        music_df[num_col].hist(bins=15)\n        plt.title(f\"Distribution of {num_col}\")\n        plt.xlabel(num_col)\n        plt.ylabel(\"Frequency\")\n        plt.tight_layout()\n    else:\n        print(\"\\nNo numeric columns found for visualization\")\n        # Create a pie chart with sample data for testing\n        plt.figure(figsize=(8, 8))\n        plt.pie([30, 20, 25, 15, 10], labels=[\"Category A\", \"Category B\", \"Category C\", \"Category D\", \"Category E\"])\n        plt.title(\"Sample Pie Chart\")\n        plt.tight_layout()\n    \n    print(\"\\nAnalysis completed successfully\")\nexcept Exception as e:\n    print(f\"Error during analysis: {e}\")"
  }'
