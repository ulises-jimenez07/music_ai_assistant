# Data Utilities

The `data_utils.py` module handles all data-related operations for the Music AI Assistant, including loading, preprocessing, and joining music datasets.

## Overview

This module is responsible for:

1. Loading music metadata and characteristics datasets
2. Joining these datasets on song IDs
3. Preprocessing the data for analysis
4. Providing schema information for code generation

## Dataset Structure

The music dataset consists of two main components:

### Music Metadata

Contains basic information about songs:
- `song_id`: Unique identifier for each song
- `artist`: Artist name
- `title`: Song title
- `lyrics_snippet`: Short excerpt from the lyrics
- `duration`: Song duration in MM:SS format
- `emotion`: Emotional classification of the song
- `genre`: Music genre
- `album`: Album name
- `release_date`: Release date of the song
- `key`: Musical key
- `tempo`: Tempo in BPM
- `loudness`: Loudness in decibels
- `time_signature`: Time signature of the song

### Music Characteristics

Contains audio features and contextual information:
- `song_id`: Unique identifier (joins with metadata)
- `explicit`: Whether the song contains explicit content
- `popularity`: Popularity score
- `energy`: Energy level (0.0 to 1.0)
- `danceability`: Danceability score (0.0 to 1.0)
- `valence`: Positiveness/mood score (0.0 to 1.0)
- `speechiness`: Presence of spoken words (0.0 to 1.0)
- `liveness`: Presence of live audience (0.0 to 1.0)
- `acousticness`: Acoustic quality (0.0 to 1.0)
- `instrumentalness`: Instrumental vs. vocal content (0.0 to 1.0)

#### Usage Context Flags
Boolean indicators for different usage contexts:
- `good_for_party`
- `good_for_work_study`
- `good_for_relaxation`
- `good_for_exercise`
- `good_for_running`
- `good_for_yoga`
- `good_for_driving`
- `good_for_social`
- `good_for_morning`

#### Similar Music Recommendations
For each song, up to 3 similar songs:
- `similar_artist_1`, `similar_song_1`, `similarity_score_1`
- `similar_artist_2`, `similar_song_2`, `similarity_score_2`
- `similar_artist_3`, `similar_song_3`, `similarity_score_3`

## Key Functions

### `load_datasets()`

Loads music metadata and characteristics datasets from the local filesystem.

- **Returns**: Tuple of (metadata_df, characteristics_df)
- **Behavior**:
  - Loads CSV files from the data directory
  - Renames columns according to predefined mappings
  - Converts boolean columns to proper boolean type
  - Handles errors gracefully, logging issues

### `join_datasets(metadata_df, characteristics_df)`

Joins metadata and characteristics datasets on song_id.

- **Parameters**:
  - `metadata_df`: DataFrame containing song metadata
  - `characteristics_df`: DataFrame containing song characteristics
- **Returns**: Joined DataFrame or None if join fails
- **Behavior**:
  - Performs an inner join on song_id
  - Logs the number of rows in the joined dataset

### `preprocess_data(df)`

Preprocesses the joined dataset for analysis.

- **Parameters**:
  - `df`: Joined DataFrame to preprocess
- **Returns**: Preprocessed DataFrame
- **Behavior**:
  - Handles missing values with appropriate defaults
  - Converts numeric columns to appropriate types
  - Extracts year from release_date
  - Converts duration from MM:SS format to seconds

### `get_full_dataset()`

Convenience function that loads, joins, and preprocesses the full dataset.

- **Returns**: Preprocessed and joined DataFrame
- **Behavior**:
  - Calls load_datasets(), join_datasets(), and preprocess_data() in sequence
  - Returns the final, analysis-ready DataFrame

### `get_dataset_schema()`

Gets the schema of the datasets for documentation and code generation.

- **Returns**: Dictionary containing schema information for both datasets
- **Behavior**:
  - Extracts column names and data types
  - Includes sample data for better understanding
  - Used by the LLM service to provide context for code generation

## Data Loading Configuration

The module uses these configuration variables:

- `LOCAL_DATA_DIR`: Path to the directory containing data files
- `USE_LOCAL_DATA`: Boolean flag to use local data (default: True)
- `METADATA_BLOB_NAME`: Filename for metadata CSV
- `CHARACTERISTICS_BLOB_NAME`: Filename for characteristics CSV

## Column Mappings

The module defines mappings between the original CSV column names and the standardized column names used in the application:

- `METADATA_COLUMNS`: Mapping for metadata columns
- `CHARACTERISTICS_COLUMNS`: Mapping for characteristics columns

These mappings ensure consistent column naming throughout the application, regardless of changes to the source data files.
