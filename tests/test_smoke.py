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
import model_utils as mu


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
    """Rows are inserted into the songs table correctly and return count is accurate."""
    df = _sample_df()
    inserted = dbu.write_songs(mem_conn, df)
    assert inserted == 2
    result = dbu.read_songs(mem_conn, limit=10)
    assert len(result) == 2
    assert "blues" in result["genre"].values
    assert "classical" in result["genre"].values


def test_write_songs_repeated_inserts_return_value(mem_conn):
    """write_songs returns the actual number of newly inserted rows, ignoring duplicates."""
    df = _sample_df()
    inserted_first = dbu.write_songs(mem_conn, df)
    assert inserted_first == 2

    # Repeated insert of identical rows returns 0
    inserted_second = dbu.write_songs(mem_conn, df)
    assert inserted_second == 0

    # Insert batch with 1 existing duplicate and 1 brand-new row
    mixed_df = pd.DataFrame([
        {
            "filename": "blues.00000.wav",  # existing duplicate
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
            "filename": "disco.00002.wav",  # brand new
            "genre": "disco",
            "tempo": 125.0,
            "chroma_stft_mean": 0.38,
            "rms_mean": 0.14,
            "spectral_centroid_mean": 2100.0,
            "spectral_bandwidth_mean": 2200.0,
            "rolloff_mean": 4000.0,
            "zero_crossing_rate_mean": 0.10,
            "mfcc1_mean": -180.0,
            "mfcc2_mean": 105.0,
            "mfcc3_mean": -4.0,
            "mfcc4_mean": 18.0,
            "mfcc5_mean": -9.0,
        },
    ])
    inserted_third = dbu.write_songs(mem_conn, mixed_df)
    assert inserted_third == 1
    total_songs = dbu.read_songs(mem_conn, limit=100)
    assert len(total_songs) == 3


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
    """Inserting the same filename twice does not create a duplicate row and returns 0."""
    df = _sample_df()
    cnt1 = dbu.write_songs(mem_conn, df)
    assert cnt1 == 2
    cnt2 = dbu.write_songs(mem_conn, df)  # second insert — should return 0
    assert cnt2 == 0
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


def test_select_best_model_uses_cv_not_test_results():
    """
    Regression test: verify select_best_model() chooses strictly via CV macro-F1,
    excluding baseline models, and does NOT select a model based on held-out test scores.
    Ensures that an implementation selecting via test results would fail.
    """
    # Scenario:
    # Model A: High CV (0.75), Moderate Test (0.71)
    # Model B: Moderate CV (0.69), High Test (0.84) -> would win if test set leaked!
    cv_scores = {
        "Baseline (Majority)": {"mean_f1": 0.018},
        "Model A (Higher CV)": {"mean_f1": 0.750},
        "Model B (Higher Test)": {"mean_f1": 0.690},
    }
    test_scores = {
        "Baseline (Majority)": {"macro_f1": 0.018},
        "Model A (Higher CV)": {"macro_f1": 0.710},
        "Model B (Higher Test)": {"macro_f1": 0.840},
    }

    # Proper implementation must select Model A based on training CV
    winner = mu.select_best_model(cv_scores, test_results=test_scores)
    assert winner == "Model A (Higher CV)"
    assert winner != "Model B (Higher Test)"
    assert winner != "Baseline (Majority)"

    # Demonstrate that an erroneous selector maximizing held-out test macro F1 picks Model B:
    def flawed_test_selector(cv_res, test_res):
        candidates = {k: v for k, v in test_res.items() if k != "Baseline (Majority)"}
        return max(candidates, key=lambda k: candidates[k]["macro_f1"])

    flawed_choice = flawed_test_selector(cv_scores, test_scores)
    assert flawed_choice == "Model B (Higher Test)"
    assert winner != flawed_choice


def test_select_best_model_baseline_ineligible():
    """Verify that baseline models can never be selected even with artificially high scores."""
    cv_scores = {
        "Baseline (Majority)": {"mean_f1": 0.999},
        "Candidate Model": {"mean_f1": 0.750},
    }
    winner = mu.select_best_model(cv_scores)
    assert winner == "Candidate Model"


def test_select_best_model_on_gtzan_cv_results():
    """Verify select_best_model on actual GTZAN cross-validation results selects SVM (RBF)."""
    gtzan_cv = {
        "Baseline (Majority)": {"mean_f1": 0.0182, "std_f1": 0.0000},
        "Logistic Regression": {"mean_f1": 0.7167, "std_f1": 0.0206},
        "SVM (RBF)": {"mean_f1": 0.7481, "std_f1": 0.0195},
        "Random Forest": {"mean_f1": 0.6973, "std_f1": 0.0077},
    }
    winner = mu.select_best_model(gtzan_cv)
    assert winner == "SVM (RBF)"


def test_gtzan_feature_schema():
    """Verify that 60 raw columns yield exactly 57 predictive audio features."""
    raw_cols = [
        "filename", "length",
        "chroma_stft_mean", "chroma_stft_var", "rms_mean", "rms_var",
        "spectral_centroid_mean", "spectral_centroid_var",
        "spectral_bandwidth_mean", "spectral_bandwidth_var",
        "rolloff_mean", "rolloff_var", "zero_crossing_rate_mean", "zero_crossing_rate_var",
        "harmony_mean", "harmony_var", "perceptr_mean", "perceptr_var", "tempo"
    ] + [f"mfcc{i}_{stat}" for i in range(1, 21) for stat in ["mean", "var"]] + ["label"]

    assert len(raw_cols) == 60
    metadata_cols = ["filename", "length"]
    target_col = "label"
    feature_cols = [c for c in raw_cols if c not in metadata_cols and c != target_col]
    assert len(feature_cols) == 57
    assert "length" not in feature_cols
    assert "filename" not in feature_cols
    assert "label" not in feature_cols

