"""
database_utils.py
-----------------
SQLite helper functions for the IDC409 Music Genre Classification project.

This module provides thin, readable wrappers around Python's built-in sqlite3
module.  Each function is intentionally short so that the notebook can call
them without burying the SQL logic, while still keeping the notebook cells
clean.

Schema
------
songs
    id          INTEGER PRIMARY KEY AUTOINCREMENT
    filename    TEXT    UNIQUE NOT NULL   -- e.g. "blues.00000.wav"
    genre       TEXT    NOT NULL          -- ground-truth label
    tempo       REAL
    chroma_stft_mean  REAL
    rms_mean          REAL
    spectral_centroid_mean REAL
    spectral_bandwidth_mean REAL
    rolloff_mean      REAL
    zero_crossing_rate_mean REAL
    mfcc1_mean        REAL
    mfcc2_mean        REAL
    mfcc3_mean        REAL
    mfcc4_mean        REAL
    mfcc5_mean        REAL

Only a representative subset of features is stored so that the database
remains readable and easy to query during a viva demonstration.
"""

import sqlite3
import pandas as pd


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def get_connection(db_path: str) -> sqlite3.Connection:
    """Return a sqlite3 connection with row_factory set for dict-like access."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

CREATE_SONGS_TABLE = """
CREATE TABLE IF NOT EXISTS songs (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    filename                 TEXT    UNIQUE NOT NULL,
    genre                    TEXT    NOT NULL,
    tempo                    REAL,
    chroma_stft_mean         REAL,
    rms_mean                 REAL,
    spectral_centroid_mean   REAL,
    spectral_bandwidth_mean  REAL,
    rolloff_mean             REAL,
    zero_crossing_rate_mean  REAL,
    mfcc1_mean               REAL,
    mfcc2_mean               REAL,
    mfcc3_mean               REAL,
    mfcc4_mean               REAL,
    mfcc5_mean               REAL
);
"""

CREATE_MODEL_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS model_results (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name   TEXT    NOT NULL,
    feature_set  TEXT    NOT NULL,   -- 'full' or 'reduced'
    accuracy     REAL,
    macro_f1     REAL,
    notes        TEXT
);
"""


def create_tables(conn: sqlite3.Connection) -> None:
    """Create tables if they do not already exist."""
    with conn:
        conn.execute(CREATE_SONGS_TABLE)
        conn.execute(CREATE_MODEL_RESULTS_TABLE)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

SONGS_COLUMNS = [
    "filename", "genre", "tempo",
    "chroma_stft_mean", "rms_mean", "spectral_centroid_mean",
    "spectral_bandwidth_mean", "rolloff_mean", "zero_crossing_rate_mean",
    "mfcc1_mean", "mfcc2_mean", "mfcc3_mean", "mfcc4_mean", "mfcc5_mean",
]

INSERT_SONG = """
INSERT OR IGNORE INTO songs
    (filename, genre, tempo, chroma_stft_mean, rms_mean,
     spectral_centroid_mean, spectral_bandwidth_mean,
     rolloff_mean, zero_crossing_rate_mean,
     mfcc1_mean, mfcc2_mean, mfcc3_mean, mfcc4_mean, mfcc5_mean)
VALUES
    (:filename, :genre, :tempo, :chroma_stft_mean, :rms_mean,
     :spectral_centroid_mean, :spectral_bandwidth_mean,
     :rolloff_mean, :zero_crossing_rate_mean,
     :mfcc1_mean, :mfcc2_mean, :mfcc3_mean, :mfcc4_mean, :mfcc5_mean);
"""


def write_songs(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """
    Insert rows from *df* into the songs table.

    Only the columns listed in SONGS_COLUMNS are written; any extra columns
    in *df* are ignored.  Rows with duplicate filenames are skipped
    (INSERT OR IGNORE).

    Returns the actual number of records inserted (excluding ignored duplicates).
    """
    subset = df[SONGS_COLUMNS].copy()
    records = subset.to_dict(orient="records")
    before_changes = conn.total_changes
    with conn:
        conn.executemany(INSERT_SONG, records)
    return conn.total_changes - before_changes


def write_model_result(
    conn: sqlite3.Connection,
    model_name: str,
    feature_set: str,
    accuracy: float,
    macro_f1: float,
    notes: str = "",
) -> None:
    """Insert one model-evaluation result row."""
    sql = """
    INSERT INTO model_results (model_name, feature_set, accuracy, macro_f1, notes)
    VALUES (?, ?, ?, ?, ?);
    """
    with conn:
        conn.execute(sql, (model_name, feature_set, accuracy, macro_f1, notes))


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def read_songs(conn: sqlite3.Connection, limit: int = 10) -> pd.DataFrame:
    """Return *limit* rows from the songs table as a DataFrame."""
    sql = "SELECT * FROM songs LIMIT ?;"
    return pd.read_sql_query(sql, conn, params=(limit,))


def read_songs_by_genre(conn: sqlite3.Connection, genre: str) -> pd.DataFrame:
    """Return all songs matching *genre* (case-insensitive)."""
    sql = "SELECT * FROM songs WHERE LOWER(genre) = LOWER(?);"
    return pd.read_sql_query(sql, conn, params=(genre,))


def read_model_results(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return all model evaluation results as a DataFrame."""
    sql = "SELECT * FROM model_results ORDER BY macro_f1 DESC;"
    return pd.read_sql_query(sql, conn)


def count_by_genre(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return a summary of song counts per genre."""
    sql = """
    SELECT genre, COUNT(*) AS song_count
    FROM songs
    GROUP BY genre
    ORDER BY song_count DESC;
    """
    return pd.read_sql_query(sql, conn)


# ---------------------------------------------------------------------------
# Modify
# ---------------------------------------------------------------------------

def update_genre(
    conn: sqlite3.Connection, filename: str, new_genre: str
) -> int:
    """
    Update the genre label for a single song identified by *filename*.

    Returns the number of rows changed (0 if the filename was not found).
    This demonstrates the SQL UPDATE operation required by the course.
    """
    sql = "UPDATE songs SET genre = ? WHERE filename = ?;"
    with conn:
        cursor = conn.execute(sql, (new_genre, filename))
    return cursor.rowcount


def verify_update(conn: sqlite3.Connection, filename: str) -> pd.DataFrame:
    """Read back the updated record to verify the change persisted."""
    sql = "SELECT id, filename, genre, tempo, mfcc1_mean FROM songs WHERE filename = ?;"
    return pd.read_sql_query(sql, conn, params=(filename,))
