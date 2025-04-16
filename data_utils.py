"""
Data utilities for the Music AI Assistant.
Handles loading, preprocessing, and joining of music datasets.
"""

import logging
import os
from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
)

import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


METADATA_BLOB_NAME = "music_metadata.csv"
CHARACTERISTICS_BLOB_NAME = "music_characteristics.csv"
LOCAL_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
USE_LOCAL_DATA = os.environ.get("USE_LOCAL_DATA", "True").lower() == "true"


# CSV column mappings
METADATA_COLUMNS = {
    "ID": "song_id",
    "Artist(s)": "artist",
    "song": "title",
    "text": "lyrics_snippet",
    "Length": "duration",
    "emotion": "emotion",
    "Genre": "genre",
    "Album": "album",
    "Release Date": "release_date",
    "Key": "key",
    "Tempo": "tempo",
    "Loudness (db)": "loudness",
    "Time signature": "time_signature",
}

CHARACTERISTICS_COLUMNS = {
    "ID": "song_id",
    "Explicit": "explicit",
    "Popularity": "popularity",
    "Energy": "energy",
    "Danceability": "danceability",
    "Positiveness": "valence",
    "Speechiness": "speechiness",
    "Liveness": "liveness",
    "Acousticness": "acousticness",
    "Instrumentalness": "instrumentalness",
    "Good for Party": "good_for_party",
    "Good for Work/Study": "good_for_work_study",
    "Good for Relaxation/Meditation": "good_for_relaxation",
    "Good for Exercise": "good_for_exercise",
    "Good for Running": "good_for_running",
    "Good for Yoga/Stretching": "good_for_yoga",
    "Good for Driving": "good_for_driving",
    "Good for Social Gatherings": "good_for_social",
    "Good for Morning Routine": "good_for_morning",
    "Similar Artist 1": "similar_artist_1",
    "Similar Song 1": "similar_song_1",
    "Similarity Score 1": "similarity_score_1",
    "Similar Artist 2": "similar_artist_2",
    "Similar Song 2": "similar_song_2",
    "Similarity Score 2": "similarity_score_2",
    "Similar Artist 3": "similar_artist_3",
    "Similar Song 3": "similar_song_3",
    "Similarity Score 3": "similarity_score_3",
}


def load_datasets() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load music metadata and characteristics datasets from local filesystem.

    Returns:
        Tuple of (metadata_df, characteristics_df)
    """
    metadata_path = os.path.join(LOCAL_DATA_DIR, "music_metadata.csv")
    characteristics_path = os.path.join(LOCAL_DATA_DIR, "music_characteristics.csv")

    # Load metadata
    try:
        metadata_df = pd.read_csv(metadata_path)
        # Rename columns according to mapping
        metadata_df = metadata_df.rename(columns=METADATA_COLUMNS)
        logger.info("Loaded metadata with %s rows", len(metadata_df))
    except Exception as e:
        logger.error("Error loading metadata: %s", str(e))
        metadata_df = None

    # Load characteristics
    try:
        characteristics_df = pd.read_csv(characteristics_path)
        # Rename columns according to mapping
        characteristics_df = characteristics_df.rename(columns=CHARACTERISTICS_COLUMNS)
        # Convert boolean columns to proper boolean type
        boolean_columns = [
            "good_for_party",
            "good_for_work_study",
            "good_for_relaxation",
            "good_for_exercise",
            "good_for_running",
            "good_for_yoga",
            "good_for_driving",
            "good_for_social",
            "good_for_morning",
        ]
        for col in boolean_columns:
            if col in characteristics_df.columns:
                characteristics_df[col] = characteristics_df[col].astype(bool)

        logger.info("Loaded characteristics with %s rows", len(characteristics_df))
    except Exception as e:
        logger.error("Error loading characteristics: %s", str(e))
        characteristics_df = None

    return metadata_df, characteristics_df


def join_datasets(metadata_df: pd.DataFrame, characteristics_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Join metadata and characteristics datasets on song_id.

    Args:
        metadata_df: DataFrame containing song metadata
        characteristics_df: DataFrame containing song characteristics

    Returns:
        Joined DataFrame or None if join fails
    """
    try:
        if metadata_df is None or characteristics_df is None:
            return None

        # Join on song_id
        joined_df = pd.merge(metadata_df, characteristics_df, on="song_id", how="inner")
        logger.info("Joined dataset has %s rows", len(joined_df))
        return joined_df
    except Exception as e:
        logger.error("Error joining datasets: %s", str(e))
        return None


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the joined dataset for analysis.

    Args:
        df: Joined DataFrame to preprocess

    Returns:
        Preprocessed DataFrame
    """
    if df is None:
        return None

    # Make a copy to avoid modifying the original
    processed_df = df.copy()

    # Handle missing values
    processed_df = processed_df.fillna(
        {
            "lyrics_snippet": "",
            "genre": "Unknown",
            "emotion": "Unknown",
            "album": "Unknown",
            "key": "Unknown",
            "release_date": "Unknown",
        }
    )

    # Convert numeric columns to appropriate types
    numeric_cols = [
        "popularity",
        "energy",
        "danceability",
        "tempo",
        "valence",
        "acousticness",
        "instrumentalness",
        "liveness",
        "speechiness",
        "similarity_score_1",
        "similarity_score_2",
        "similarity_score_3",
    ]

    for col in numeric_cols:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors="coerce")

    # Extract year from release_date if present
    if "release_date" in processed_df.columns:
        # Try to extract year from release_date
        processed_df["release_year"] = processed_df["release_date"].str.extract(r"(\d{4})")
        processed_df["release_year"] = (
            pd.to_numeric(processed_df["release_year"], errors="coerce").fillna(0).astype(int)
        )

    # Convert duration to seconds if present (format: MM:SS)
    if "duration" in processed_df.columns:

        def convert_duration(duration_str):
            try:
                if pd.isna(duration_str):
                    return None
                parts = duration_str.split(":")
                if len(parts) == 2:
                    return int(parts[0]) * 60 + int(parts[1])
                if len(parts) == 3:
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                return None
            except Exception:
                return None

        processed_df["duration_sec"] = processed_df["duration"].apply(convert_duration)

    return processed_df


def get_full_dataset() -> pd.DataFrame:
    """
    Load, join, and preprocess the full dataset.

    By default, uses local files for testing and development.

    Returns:
        Preprocessed and joined DataFrame
    """
    metadata_df, characteristics_df = load_datasets()
    joined_df = join_datasets(metadata_df, characteristics_df)
    return preprocess_data(joined_df)


def get_dataset_schema() -> Dict[str, Dict[str, Any]]:
    """
    Get the schema of the datasets for documentation and code generation.

    Returns:
        Dictionary containing schema information for both datasets
    """
    metadata_df, characteristics_df = load_datasets()

    schema: Dict[str, Dict[str, Any]] = {"metadata": {}, "characteristics": {}}

    if metadata_df is not None:
        schema["metadata"] = {
            "columns": list(metadata_df.columns),
            "dtypes": {col: str(metadata_df[col].dtype) for col in metadata_df.columns},
            "sample": metadata_df.head(3).to_dict(orient="records"),
        }

    if characteristics_df is not None:
        schema["characteristics"] = {
            "columns": list(characteristics_df.columns),
            "dtypes": {col: str(characteristics_df[col].dtype) for col in characteristics_df.columns},
            "sample": characteristics_df.head(3).to_dict(orient="records"),
        }

    return schema


if __name__ == "__main__":
    # Test data loading and preprocessing
    full_df = get_full_dataset()
    if full_df is not None:
        print("Successfully loaded and preprocessed dataset with %s rows" % len(full_df))
        print("Columns: %s" % full_df.columns.tolist())
        print("\nSample data:")
        print(full_df.head(2))
        print(get_dataset_schema())
    else:
        print("Failed to load and preprocess dataset")
