"""
test_smoke.py
-------------
Quick smoke tests for the IDC409 Music Genre Classification project.

These tests verify that the environment and database utilities work correctly
WITHOUT requiring the actual GTZAN dataset to be present.

Run from the project root:
    python -m pytest tests/test_smoke.py -v
"""

import sys
import os
import sqlite3
import tempfile

import numpy as np
import pandas as pd
import pytest

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import database_utils as dbu


# ---------------------------------------------------------------------------
# Environment smoke tests
# ---------------------------------------------------------------------------

def test_numpy_available():
    """NumPy can be imported and a basic array operation works."""
    arr = np.array([1.0, 2.0, 3.0])
    assert arr.mean() == 2.0


def test_pandas_available():
    """pandas can be imported and a simple DataFrame is created."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert df.shape == (3, 2)


def test_sklearn_available():
    """scikit-learn can be imported."""
    from sklearn.dummy import DummyClassifier
    assert DummyClassifier is not None


def test_matplotlib_available():
    """Matplotlib can be imported (non-interactive backend)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    assert plt is not None


def test_seaborn_available():
    """Seaborn can be imported."""
    import seaborn as sns
    assert sns.__version__ is not None


def test_sqlite3_available():
    """Python's sqlite3 module is available."""
    conn = sqlite3.connect(":memory:")
    conn.close()


# ---------------------------------------------------------------------------
# Database utility tests (using an in-memory SQLite database)
# ---------------------------------------------------------------------------

@pytest.fixture
def mem_conn():
    """Provide an in-memory SQLite connection for testing."""
    conn = dbu.get_connection(":memory:")
    dbu.create_tables(conn)
    yield conn
    conn.close()


def _sample_df():
    """Return a tiny DataFrame that mirrors the songs table columns."""
    return pd.DataFrame(
        [
            {
                "filename": "blues.00000.wav",
                "genre": "blues",
                "tempo": 120.0,
                "chroma_stft_mean": 0.35,
                "rms_mean": 0.12,
                "spectral_centroid_mean": 1800.0,
                "spectral_bandwidth_mean": 2000.0,
                "rolloff_mean": 3500.0,
                "zero_crossing_rate_mean": 0.08,
                "mfcc1_mean": -200.0,
                "mfcc2_mean": 110.0,
                "mfcc3_mean": -5.0,
                "mfcc4_mean": 20.0,
                "mfcc5_mean": -10.0,
            },
            {
                "filename": "classical.00001.wav",
                "genre": "classical",
                "tempo": 90.0,
                "chroma_stft_mean": 0.40,
                "rms_mean": 0.08,
                "spectral_centroid_mean": 1200.0,
                "spectral_bandwidth_mean": 1500.0,
                "rolloff_mean": 2500.0,
                "zero_crossing_rate_mean": 0.05,
                "mfcc1_mean": -150.0,
                "mfcc2_mean": 90.0,
                "mfcc3_mean": -2.0,
                "mfcc4_mean": 15.0,
                "mfcc5_mean": -8.0,
            },
        ]
    )


def test_create_tables(mem_conn):
    """Tables are created without error."""
    # Check songs table exists
    cursor = mem_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='songs';"
    )
    assert cursor.fetchone() is not None

    # Check model_results table exists
    cursor = mem_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='model_results';"
    )
    assert cursor.fetchone() is not None


def test_write_songs(mem_conn):
    """Rows are inserted into the songs table correctly."""
    df = _sample_df()
    dbu.write_songs(mem_conn, df)
    result = dbu.read_songs(mem_conn, limit=10)
    assert len(result) == 2
    assert "blues" in result["genre"].values
    assert "classical" in result["genre"].values


def test_read_songs_by_genre(mem_conn):
    """read_songs_by_genre returns only the requested genre."""
    df = _sample_df()
    dbu.write_songs(mem_conn, df)
    result = dbu.read_songs_by_genre(mem_conn, "blues")
    assert len(result) == 1
    assert result.iloc[0]["genre"] == "blues"


def test_update_genre(mem_conn):
    """update_genre changes the genre field and the change is readable."""
    df = _sample_df()
    dbu.write_songs(mem_conn, df)

    rows_changed = dbu.update_genre(mem_conn, "blues.00000.wav", "jazz")
    assert rows_changed == 1

    verified = dbu.verify_update(mem_conn, "blues.00000.wav")
    assert verified.iloc[0]["genre"] == "jazz"


def test_duplicate_insert_ignored(mem_conn):
    """Inserting the same filename twice does not create a duplicate row."""
    df = _sample_df()
    dbu.write_songs(mem_conn, df)
    dbu.write_songs(mem_conn, df)  # second insert — should be ignored
    result = dbu.read_songs(mem_conn, limit=100)
    assert len(result) == 2  # still 2 rows


def test_write_model_result(mem_conn):
    """Model results can be written and read back."""
    dbu.write_model_result(
        mem_conn,
        model_name="RandomForest",
        feature_set="full",
        accuracy=0.82,
        macro_f1=0.81,
        notes="Test run",
    )
    results = dbu.read_model_results(mem_conn)
    assert len(results) == 1
    assert results.iloc[0]["model_name"] == "RandomForest"


def test_count_by_genre(mem_conn):
    """count_by_genre returns correct counts."""
    df = _sample_df()
    dbu.write_songs(mem_conn, df)
    counts = dbu.count_by_genre(mem_conn)
    assert set(counts["genre"].values) == {"blues", "classical"}
    # Each genre has exactly 1 song in our sample
    assert (counts["song_count"] == 1).all()


# ---------------------------------------------------------------------------
# ML-pipeline sanity tests (no dataset needed)
# ---------------------------------------------------------------------------

def test_sklearn_pipeline_imports():
    """Key scikit-learn components can be imported."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        classification_report,
        confusion_matrix,
    )
    assert True  # if we get here, all imports worked


def test_dummy_classifier_trains():
    """DummyClassifier trains and predicts on a tiny synthetic dataset."""
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import accuracy_score

    X = np.random.default_rng(42).standard_normal((60, 5))
    y = np.array(["blues"] * 20 + ["rock"] * 20 + ["jazz"] * 20)

    clf = DummyClassifier(strategy="most_frequent", random_state=42)
    clf.fit(X, y)
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    assert 0.0 <= acc <= 1.0


def test_cosine_similarity_import():
    """sklearn cosine_similarity is importable (needed for recommender)."""
    from sklearn.metrics.pairwise import cosine_similarity
    mat = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    sims = cosine_similarity(mat)
    assert sims.shape == (3, 3)
