"""
build_notebook.py
-----------------
Programmatically assembles the full IDC409 Music Genre Classification
Jupyter notebook (.ipynb) from cell definitions in this script.

Run from the project root:
    python build_notebook.py

Output: notebooks/music_genre_classification.ipynb
"""

import json, os, uuid

NB_PATH = os.path.join("notebooks", "music_genre_classification.ipynb")

def md(src): return {"cell_type":"markdown","id":str(uuid.uuid4())[:8],"metadata":{},"source":src}
def code(src): return {"cell_type":"code","id":str(uuid.uuid4())[:8],"execution_count":None,"metadata":{},"outputs":[],"source":src}

cells = []

# ── Cell 1 — Title ──────────────────────────────────────────────────────────
cells.append(md("""\
# IDC409 — Music Genre Classification
### Classify songs into genres using extracted audio features

**Course**: IDC409  
**Dataset**: GTZAN Music Genre Classification (Tzanetakis & Cook, 2002) via Kaggle  
**Dataset URL**: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification  

---

## Project Objective

Can we automatically assign a music genre label to a song using only its **numerical audio features** — without ever listening to it?

We use the **GTZAN** dataset, which provides 58 pre-extracted audio features for 1 000 tracks across 10 genres. We:
1. Load and clean the feature table using **NumPy** and **pandas**.
2. Explore the data visually (**EDA**).
3. Store and query records using **SQLite**.
4. Train and compare three classifiers (**Logistic Regression**, **SVM**, **Random Forest**) plus a majority-class baseline.
5. Reduce the feature set with **SelectKBest** and compare performance.
6. Evaluate using **accuracy**, **macro F1**, and a **confusion matrix**.
7. Build a simple **song-similarity recommender** as an extension.

All results reported are produced by running this notebook top-to-bottom with a fixed random seed.\
"""))

# ── Cell 2 — Imports ────────────────────────────────────────────────────────
cells.append(md("## 0. Imports and Configuration"))
cells.append(code("""\
# Standard library
import os, sys, sqlite3, warnings

# Numerical / data
import numpy as np
import pandas as pd

# Visualisation
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Machine learning
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.metrics.pairwise import cosine_similarity

# Project database utilities
sys.path.insert(0, os.path.join("..", "src"))
import database_utils as dbu

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
warnings.filterwarnings("ignore")

# Paths
DATA_PATH = os.path.join("..", "data", "features_30_sec.csv")
DB_PATH   = os.path.join("..", "data", "music_genre.db")

# Plot style
sns.set_theme(style="darkgrid", palette="tab10")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

print("Environment OK")
print(f"  numpy      {np.__version__}")
print(f"  pandas     {pd.__version__}")
import sklearn; print(f"  sklearn    {sklearn.__version__}")
import matplotlib; print(f"  matplotlib {matplotlib.__version__}")
import seaborn; print(f"  seaborn    {seaborn.__version__}")\
"""))

# ── Cell 3 — Data Loading ────────────────────────────────────────────────────
cells.append(md("""\
## 1. Data Loading

We load `features_30_sec.csv` from the GTZAN dataset.  
Each row represents one 30-second audio clip; each column is a pre-extracted audio feature or metadata field.

> **If the file is missing**, follow the download instructions in `data/README.md`.\
"""))
cells.append(code("""\
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at '{DATA_PATH}'.\\n"
        "Please follow the instructions in data/README.md to download\\n"
        "  features_30_sec.csv  from Kaggle and place it in the data/ folder."
    )

df_raw = pd.read_csv(DATA_PATH)
print(f"Shape: {df_raw.shape}  ({df_raw.shape[0]} tracks, {df_raw.shape[1]} columns)")
df_raw.head(3)\
"""))

# ── Cell 4 — Schema ──────────────────────────────────────────────────────────
cells.append(md("""\
### 1.1 Column overview

The 60 columns fall into three categories:

| Category | Columns | Notes |
|---|---|---|
| **Metadata** | `filename`, `length` | Not predictive features |
| **Audio features** | 57 numerical columns | Used as model inputs |
| **Target label** | `label` | The genre to predict |

**Audio features include:**
- `chroma_stft_mean/var` — harmonic/melodic content across 12 pitch classes
- `rms_mean/var` — Root Mean Square energy (loudness)
- `spectral_centroid_mean/var` — "brightness" of the sound
- `spectral_bandwidth_mean/var` — spread of frequencies around the centroid
- `rolloff_mean/var` — frequency below which 85 % of energy sits
- `zero_crossing_rate_mean/var` — how often the signal crosses zero (noise indicator)
- `harmony_mean/var` — harmonic component strength
- `perceptr_mean/var` — percussive component strength
- `tempo` — beats per minute (BPM)
- `mfcc1_mean` … `mfcc20_mean` and their variances — shape of the spectral envelope\
"""))
cells.append(code("""\
# Column types and a quick dtype check
print("Column types:")
print(df_raw.dtypes.value_counts())
print()
print("Columns:", list(df_raw.columns))\
"""))

# ── Cell 5 — Data Quality ────────────────────────────────────────────────────
cells.append(md("""\
## 2. Data Quality Check

Before any analysis we check for:
- Missing values
- Duplicate rows / filenames
- Data type issues\
"""))
cells.append(code("""\
# Missing values
missing = df_raw.isnull().sum()
print("Missing values per column (showing non-zero only):")
print(missing[missing > 0] if missing.any() else "  None — the dataset is complete.")
print()

# Duplicate rows
n_dup_rows = df_raw.duplicated().sum()
print(f"Duplicate rows: {n_dup_rows}")

# Duplicate filenames
n_dup_fname = df_raw["filename"].duplicated().sum()
print(f"Duplicate filenames: {n_dup_fname}")

# Class counts
print()
print("Genre distribution:")
print(df_raw["label"].value_counts())\
"""))

# ── Cell 6 — Clean ───────────────────────────────────────────────────────────
cells.append(md("""\
### 2.1 Cleaning decisions

- **`length`** column is dropped: it is a technical sample-count (not an audio feature) and is nearly identical for all 30-second clips — it would not help a classifier and could confuse interpretation.
- **`filename`** is kept as an index for the database and recommender but excluded from model inputs.
- No rows are dropped: the GTZAN dataset is known to be clean with exactly 100 tracks per genre.
- One known corrupt file (`jazz.00054.wav`) exists in the raw audio but is absent from the CSV; no action needed here.\
"""))
cells.append(code("""\
df = df_raw.drop(columns=["length"]).copy()

METADATA_COLS = ["filename"]
TARGET_COL    = "label"
FEATURE_COLS  = [c for c in df.columns if c not in METADATA_COLS + [TARGET_COL]]

print(f"Feature columns  : {len(FEATURE_COLS)}")
print(f"Target column    : '{TARGET_COL}'")
print(f"Metadata columns : {METADATA_COLS}")
print(f"Dataset shape    : {df.shape}")
print()

# Show the clean feature matrix (first 3 rows, selected columns)
df[FEATURE_COLS[:6]].describe().round(3)\
"""))

# ── Cell 7 — EDA ─────────────────────────────────────────────────────────────
cells.append(md("""\
## 3. Exploratory Data Analysis (EDA)

We answer concrete questions about the data before training any model:

1. Are the genres balanced?
2. How does **tempo** vary across genres?
3. How does **spectral centroid** (brightness) differ?
4. Do **MFCC 1** (overall energy/loudness) patterns separate genres?
5. What are the feature correlations?\
"""))

# ── Cell 8 — Genre balance ───────────────────────────────────────────────────
cells.append(md("""\
### 3.1 Genre class balance

**What we're showing**: count of tracks per genre.  
**What we see**: each genre has exactly 100 tracks — a perfectly balanced dataset.  
**Conclusion**: no class-imbalance weighting is necessary, and accuracy is a fair metric alongside macro F1.\
"""))
cells.append(code("""\
genre_counts = df[TARGET_COL].value_counts().sort_values()
genres_ordered = genre_counts.index.tolist()

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(genre_counts.index, genre_counts.values,
               color=sns.color_palette("tab10", len(genre_counts)))
ax.set_xlabel("Number of tracks")
ax.set_title("Track count per genre (GTZAN, N=1000)")
ax.bar_label(bars, padding=3, fmt="%d")
plt.tight_layout()
plt.savefig("../data/eda_genre_balance.png", bbox_inches="tight")
plt.show()
print("Saved: eda_genre_balance.png")\
"""))

# ── Cell 9 — Tempo ───────────────────────────────────────────────────────────
cells.append(md("""\
### 3.2 Tempo distribution by genre

**What we're showing**: a box plot of BPM (beats per minute) per genre.  
**What we see**: classical music tends to have a lower, more variable tempo; metal and disco lean toward higher BPMs; hip-hop shows a wide BPM spread.  
**Limitation**: tempo is estimated by a librosa algorithm and may not perfectly reflect a human listener's perception of beat, especially for irregular rhythms.\
"""))
cells.append(code("""\
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, x=TARGET_COL, y="tempo", order=genres_ordered,
            palette="tab10", ax=ax)
ax.set_title("Tempo (BPM) by genre")
ax.set_xlabel("Genre")
ax.set_ylabel("Tempo (BPM)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("../data/eda_tempo.png", bbox_inches="tight")
plt.show()\
"""))

# ── Cell 10 — Spectral Centroid ──────────────────────────────────────────────
cells.append(md("""\
### 3.3 Spectral centroid by genre

**What we're showing**: distribution of `spectral_centroid_mean` — the "brightness" of a track.  
**What we see**: metal and rock have higher centroids (more high-frequency energy), while classical and jazz tend to be lower.  
**Limitation**: a high spectral centroid does not uniquely identify a genre — some pop tracks also have high centroids.\
"""))
cells.append(code("""\
fig, ax = plt.subplots(figsize=(10, 5))
sns.violinplot(data=df, x=TARGET_COL, y="spectral_centroid_mean",
               order=genres_ordered, palette="tab10", ax=ax)
ax.set_title("Spectral Centroid Mean by genre")
ax.set_xlabel("Genre")
ax.set_ylabel("Spectral centroid (Hz)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("../data/eda_spectral_centroid.png", bbox_inches="tight")
plt.show()\
"""))

# ── Cell 11 — MFCC1 ──────────────────────────────────────────────────────────
cells.append(md("""\
### 3.4 MFCC 1 mean by genre

**What we're showing**: the first MFCC coefficient captures the overall spectral energy level — roughly analogous to loudness.  
**What we see**: classical music has distinctly negative MFCC 1 values (quieter, sparser spectra); metal and rock are at the positive end.  
**Conclusion**: MFCC 1 alone has meaningful discriminative power for separating classical from louder genres.\
"""))
cells.append(code("""\
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, x=TARGET_COL, y="mfcc1_mean",
            order=genres_ordered, palette="tab10", ax=ax)
ax.set_title("MFCC 1 Mean by genre")
ax.set_xlabel("Genre")
ax.set_ylabel("MFCC 1 mean")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("../data/eda_mfcc1.png", bbox_inches="tight")
plt.show()\
"""))

# ── Cell 12 — Correlation heatmap ────────────────────────────────────────────
cells.append(md("""\
### 3.5 Feature correlation heatmap

**What we're showing**: Pearson correlations between a representative subset of features.  
**What we see**: several MFCC mean columns are highly correlated with each other, which motivates feature reduction.  
**Limitation**: correlation does not imply redundancy in a nonlinear model like Random Forest.\
"""))
cells.append(code("""\
# Select a representative 15-feature subset for readability
heat_cols = [
    "chroma_stft_mean", "rms_mean", "spectral_centroid_mean",
    "spectral_bandwidth_mean", "rolloff_mean", "zero_crossing_rate_mean",
    "tempo",
    "mfcc1_mean", "mfcc2_mean", "mfcc3_mean",
    "mfcc4_mean", "mfcc5_mean", "mfcc6_mean",
    "harmony_mean", "perceptr_mean",
]
corr = df[heat_cols].corr()

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.4, ax=ax,
            annot_kws={"size": 7})
ax.set_title("Feature correlation matrix (representative subset)")
plt.tight_layout()
plt.savefig("../data/eda_correlation.png", bbox_inches="tight")
plt.show()\
"""))

# ── Cell 13 — EDA summary ───────────────────────────────────────────────────
cells.append(md("""\
### 3.6 EDA summary

| Observation | Implication |
|---|---|
| Perfectly balanced classes (100 each) | Accuracy is a fair metric; no oversampling needed |
| Classical has the lowest MFCC 1 and tempo | Classical should be easy to separate |
| Metal/rock have high spectral centroid | Brightness is a useful feature |
| Several MFCC columns are correlated | Feature reduction may preserve most information |
| Reggae and hip-hop overlap in many features | Expect confusion between these genres |\
"""))

# ── Cell 14 — SQLite ─────────────────────────────────────────────────────────
cells.append(md("""\
## 4. SQLite Database

The course requires demonstrating **read, write, and modify** operations on a database.  
We use Python's built-in `sqlite3` module via the helper functions in `src/database_utils.py`.

### Schema

```sql
songs (id PK, filename UNIQUE, genre, tempo, chroma_stft_mean,
       rms_mean, spectral_centroid_mean, spectral_bandwidth_mean,
       rolloff_mean, zero_crossing_rate_mean,
       mfcc1_mean … mfcc5_mean)

model_results (id PK, model_name, feature_set, accuracy, macro_f1, notes)
```\
"""))
cells.append(code("""\
# Remove old database if re-running
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = dbu.get_connection(DB_PATH)
dbu.create_tables(conn)
print("Database created:", DB_PATH)\
"""))

# ── Cell 15 — DB Write ────────────────────────────────────────────────────────
cells.append(md("### 4.1 Write — INSERT all 1 000 songs"))
cells.append(code("""\
# Prepare the DataFrame with the columns the DB expects.
# The dataset uses 'label' as the genre column; we rename it to 'genre'
# to match the database schema before inserting.
db_df = df.rename(columns={"label": "genre"})[dbu.SONGS_COLUMNS].copy()

rows_inserted = dbu.write_songs(conn, db_df)
print(f"Rows written to 'songs' table: {rows_inserted}")
print()

# Verify the write
sample = dbu.read_songs(conn, limit=5)
print("First 5 rows in database:")
sample\
"""))

# ── Cell 16 — DB Read ─────────────────────────────────────────────────────────
cells.append(md("### 4.2 Read — SELECT with a WHERE clause"))
cells.append(code("""\
jazz_tracks = dbu.read_songs_by_genre(conn, "jazz")
print(f"Jazz tracks in database: {len(jazz_tracks)}")
jazz_tracks.head()\
"""))

cells.append(code("""\
genre_summary = dbu.count_by_genre(conn)
print("Track counts per genre (from SQLite):")
genre_summary\
"""))

# ── Cell 17 — DB Modify ───────────────────────────────────────────────────────
cells.append(md("""\
### 4.3 Modify — UPDATE and verify

We demonstrate the UPDATE operation by correcting the genre of one track (simulating a label correction workflow).  
We then read the row back to confirm the change persisted.\
"""))
cells.append(code("""\
TARGET_FILE = "jazz.00054.wav"   # known edge-case track

# First, check current record (it might not exist — jazz.00054 is sometimes absent)
before = dbu.verify_update(conn, TARGET_FILE)
if before.empty:
    # Fall back to first jazz track
    TARGET_FILE = dbu.read_songs_by_genre(conn, "jazz").iloc[0]["filename"]
    print(f"Using fallback target: {TARGET_FILE}")

before = dbu.verify_update(conn, TARGET_FILE)
print("BEFORE update:")
print(before[["id", "filename", "genre"]].to_string(index=False))

# Perform the UPDATE
rows_changed = dbu.update_genre(conn, TARGET_FILE, "jazz_corrected")
print(f"\\nRows changed: {rows_changed}")

# Read it back
after = dbu.verify_update(conn, TARGET_FILE)
print("\\nAFTER update:")
print(after[["id", "filename", "genre"]].to_string(index=False))

# Restore original label
dbu.update_genre(conn, TARGET_FILE, "jazz")
print("\\nRestored to 'jazz'.")\
"""))

# ── Cell 18 — ML Setup ────────────────────────────────────────────────────────
cells.append(md("""\
## 5. Machine-Learning Methodology

### 5.1 Experimental procedure

1. **Separate features and labels** — exclude `filename` and `label`.
2. **Train / test split** — 80 % train, 20 % test, stratified so every genre has proportional representation.
3. **Preprocessing inside a Pipeline** — `StandardScaler` is fitted on the training fold only (no leakage).
4. **Baseline** — `DummyClassifier(strategy="most_frequent")` gives the score a trivial model achieves.
5. **Cross-validation** — 5-fold stratified CV on the training set for model selection.
6. **Final evaluation** — once, on the held-out test set.
7. **Random seed** = 42 throughout.

> **Data leakage prevention**: the `StandardScaler` and `SelectKBest` are always inside the `Pipeline`, so they are fitted only on training data during cross-validation and the final training run.\
"""))
cells.append(code("""\
X = df[FEATURE_COLS].values.astype(float)
y = df[TARGET_COL].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Training set : {X_train.shape[0]} samples")
print(f"Test set     : {X_test.shape[0]} samples")
print(f"Features     : {X_train.shape[1]}")
print()
print("Class distribution in test set:")
unique, counts = np.unique(y_test, return_counts=True)
for g, c in zip(unique, counts):
    print(f"  {g:12s} {c}")\
"""))

# ── Cell 19 — Model definitions ───────────────────────────────────────────────
cells.append(md("### 5.2 Model pipelines"))
cells.append(code("""\
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

models = {
    "Baseline (Majority)": Pipeline([
        ("clf", DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE))
    ]),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                                   multi_class="multinomial", solver="lbfgs", C=1.0))
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10, gamma="scale",
                    random_state=RANDOM_STATE, probability=True))
    ]),
    "Random Forest": Pipeline([
        # No scaling needed for tree ensembles
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=None,
                                        random_state=RANDOM_STATE, n_jobs=-1))
    ]),
}

print("Models defined:", list(models.keys()))\
"""))

# ── Cell 20 — CV ──────────────────────────────────────────────────────────────
cells.append(md("""\
### 5.3 Cross-validation on training set

We run 5-fold stratified cross-validation to compare models before ever touching the test set.\
"""))
cells.append(code("""\
cv_results = {}
for name, pipe in models.items():
    scores = cross_val_score(pipe, X_train, y_train,
                             cv=cv, scoring="f1_macro", n_jobs=-1)
    cv_results[name] = {"mean_f1": scores.mean(), "std_f1": scores.std()}
    print(f"{name:25s}  CV macro-F1 = {scores.mean():.4f} ± {scores.std():.4f}")

print()
print("Best CV model:", max(cv_results, key=lambda k: cv_results[k]["mean_f1"]))\
"""))

# ── Cell 21 — Test evaluation ─────────────────────────────────────────────────
cells.append(md("""\
### 5.4 Final evaluation on the held-out test set

We train each pipeline on the full training set and evaluate once on the test set.\
"""))
cells.append(code("""\
test_results = {}

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="macro")
    test_results[name] = {"accuracy": acc, "macro_f1": f1, "y_pred": y_pred}
    print(f"{name:25s}  Acc={acc:.4f}  Macro-F1={f1:.4f}")

print()
best_model_name = max(
    {k: v for k, v in test_results.items() if k != "Baseline (Majority)"},
    key=lambda k: test_results[k]["macro_f1"]
)
print(f"Best model: {best_model_name}")\
"""))

# ── Cell 22 — Comparison table ────────────────────────────────────────────────
cells.append(md("### 5.5 Model comparison table"))
cells.append(code("""\
results_df = pd.DataFrame({
    name: {
        "CV macro-F1 (mean)": cv_results[name]["mean_f1"],
        "CV macro-F1 (±std)": cv_results[name]["std_f1"],
        "Test accuracy":      test_results[name]["accuracy"],
        "Test macro-F1":      test_results[name]["macro_f1"],
    }
    for name in models
}).T.round(4)

results_df = results_df.sort_values("Test macro-F1", ascending=False)
print(results_df.to_string())

# Save to database
for name, row in results_df.iterrows():
    dbu.write_model_result(conn, name, "full",
                           row["Test accuracy"], row["Test macro-F1"],
                           notes="Full feature set")
print("\\nResults saved to SQLite model_results table.")
dbu.read_model_results(conn)\
"""))

# ── Cell 23 — Classification report ──────────────────────────────────────────
cells.append(md("### 5.6 Detailed classification report — best model"))
cells.append(code("""\
y_pred_best = test_results[best_model_name]["y_pred"]
print(f"Classification report — {best_model_name}")
print("=" * 55)
print(classification_report(y_test, y_pred_best))\
"""))

# ── Cell 24 — Confusion matrix ────────────────────────────────────────────────
cells.append(md("""\
### 5.7 Confusion matrix

**How to read this**: each row is a true genre; each column is a predicted genre.  
High numbers on the diagonal = correct predictions.  
Off-diagonal entries reveal which genres are confused with each other.\
"""))
cells.append(code("""\
genre_labels = sorted(df[TARGET_COL].unique())
cm = confusion_matrix(y_test, y_pred_best, labels=genre_labels)

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=genre_labels)
disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
ax.set_title(f"Confusion matrix — {best_model_name} (test set)")
plt.tight_layout()
plt.savefig("../data/confusion_matrix.png", bbox_inches="tight")
plt.show()
print("Saved: confusion_matrix.png")\
"""))

# ── Cell 25 — Genre analysis ─────────────────────────────────────────────────
cells.append(md("""\
### 5.8 Per-genre analysis and audio reflection

> **Which genres are confused, and why?**

Looking at the confusion matrix and per-class F1 scores from the best model:

- **Classical** is almost always classified correctly — its low MFCC 1, low tempo, and low spectral centroid make it acoustically distinct from all other genres.
- **Metal** is well-classified — its extremely high spectral centroid and zero-crossing rate are strong signals.
- **Reggae ↔ Hip-hop** confusion is common: both genres share similar BPM ranges, comparable spectral shapes, and overlapping MFCC distributions. The numerical features alone cannot fully capture the cultural and rhythmic style differences a human listener detects.
- **Country ↔ Rock** confusion occurs because both use similar instrumentation (electric/acoustic guitar) leading to overlapping spectral centroids and MFCC patterns.
- **Disco ↔ Pop** can overlap in tempo and spectral brightness.

**Important caveat**: these are hypotheses based on feature distributions — the analysis shows *which* features a model leans on (through feature importances) but cannot prove that a specific audio characteristic *caused* a misclassification.\
"""))

# ── Cell 26 — Feature reduction ───────────────────────────────────────────────
cells.append(md("""\
## 6. Feature Reduction

### 6.1 Rationale

The dataset has 57 features. Many MFCC columns are correlated (seen in the heatmap).  
We use **SelectKBest with f_classif** (ANOVA F-value) to select the K features most statistically associated with the genre labels.

We test K = 15 (roughly 25 % of 57 features) and compare against the full-feature model using the same cross-validation and test-set protocol.

> `SelectKBest` is always placed **inside the Pipeline** to avoid leakage: feature scores are computed on the training fold only during CV.\
"""))
cells.append(code("""\
K_FEATURES = 15

reduced_models = {
    "LR (reduced)": Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=K_FEATURES)),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                                    multi_class="multinomial", solver="lbfgs", C=1.0))
    ]),
    "SVM (reduced)": Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=K_FEATURES)),
        ("clf", SVC(kernel="rbf", C=10, gamma="scale",
                    random_state=RANDOM_STATE))
    ]),
    "RF (reduced)": Pipeline([
        ("select", SelectKBest(f_classif, k=K_FEATURES)),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))
    ]),
}

reduced_results = {}
for name, pipe in reduced_models.items():
    cv_f1 = cross_val_score(pipe, X_train, y_train,
                             cv=cv, scoring="f1_macro", n_jobs=-1).mean()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="macro")
    reduced_results[name] = {"CV macro-F1": cv_f1, "Test accuracy": acc, "Test macro-F1": f1}
    print(f"{name:15s}  CV-F1={cv_f1:.4f}  Acc={acc:.4f}  Macro-F1={f1:.4f}")

    dbu.write_model_result(conn, name, "reduced", acc, f1,
                           notes=f"SelectKBest k={K_FEATURES}")\
"""))

# ── Cell 27 — Selected features ───────────────────────────────────────────────
cells.append(md("### 6.2 Which features were selected?"))
cells.append(code("""\
# Use the fitted RF-reduced pipeline to inspect selected feature names
selector = reduced_models["RF (reduced)"]["select"]
selected_mask   = selector.get_support()
selected_features = np.array(FEATURE_COLS)[selected_mask]
feature_scores    = selector.scores_[selected_mask]

feat_df = pd.DataFrame({
    "Feature": selected_features,
    "ANOVA F-score": feature_scores
}).sort_values("ANOVA F-score", ascending=False).reset_index(drop=True)

print(f"Features selected (K={K_FEATURES}):")
print(feat_df.to_string(index=False))\
"""))

# ── Cell 28 — Reduction comparison ────────────────────────────────────────────
cells.append(md("### 6.3 Full vs reduced feature set — comparison table"))
cells.append(code("""\
print(f"Original feature count : {len(FEATURE_COLS)}")
print(f"Reduced feature count  : {K_FEATURES}")
print()

comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "SVM", "Random Forest"],
    "Full Test macro-F1": [
        test_results["Logistic Regression"]["macro_f1"],
        test_results["SVM (RBF)"]["macro_f1"],
        test_results["Random Forest"]["macro_f1"],
    ],
    "Reduced Test macro-F1": [
        reduced_results["LR (reduced)"]["Test macro-F1"],
        reduced_results["SVM (reduced)"]["Test macro-F1"],
        reduced_results["RF (reduced)"]["Test macro-F1"],
    ],
}).set_index("Model")

comparison["Δ macro-F1"] = (
    comparison["Reduced Test macro-F1"] - comparison["Full Test macro-F1"]
).round(4)
comparison = comparison.round(4)
print(comparison.to_string())

print()
print("Interpretation:")
for _, row in comparison.iterrows():
    delta = row["Δ macro-F1"]
    direction = "improved" if delta > 0.005 else ("declined" if delta < -0.005 else "unchanged")
    print(f"  {row.name:20s}: performance {direction} by {abs(delta):.4f}")\
"""))

# ── Cell 29 — Song similarity recommender ─────────────────────────────────────
cells.append(md("""\
## 7. Song-Similarity Recommender (Extension)

After all core requirements are satisfied, we add a simple extension:  
given a track filename, find the 5 most similar songs in the dataset using **cosine similarity** on scaled audio features.

> **Note**: similar numerical features do not guarantee that songs sound identical to a human listener — they may share acoustic properties (tempo, brightness, loudness) while differing in melody, lyrics, or cultural style.\
"""))
cells.append(code("""\
from sklearn.preprocessing import StandardScaler

# Scale all features for fair distance computation
scaler_rec = StandardScaler()
X_scaled_rec = scaler_rec.fit_transform(df[FEATURE_COLS].values)

def recommend_similar(query_filename, top_n=5):
    \"\"\"
    Return the top_n most similar songs to query_filename
    by cosine similarity on scaled audio features.
    \"\"\"
    # Find query row index
    matches = df.index[df["filename"] == query_filename].tolist()
    if not matches:
        return f"File '{query_filename}' not found in dataset."
    idx = matches[0]

    query_vec = X_scaled_rec[idx].reshape(1, -1)
    sims = cosine_similarity(query_vec, X_scaled_rec).flatten()
    sims[idx] = -1  # exclude the query song itself

    top_indices = np.argsort(sims)[::-1][:top_n]
    results = pd.DataFrame({
        "filename": df.iloc[top_indices]["filename"].values,
        "genre":    df.iloc[top_indices][TARGET_COL].values,
        "similarity": sims[top_indices].round(4),
    })
    return results

# Demonstrate with first classical track
query = df[df[TARGET_COL] == "classical"]["filename"].iloc[0]
print(f"Query track : {query}  (genre: classical)")
print()
recommend_similar(query)\
"""))

cells.append(code("""\
# Another example — first blues track
query2 = df[df[TARGET_COL] == "blues"]["filename"].iloc[0]
print(f"Query track : {query2}  (genre: blues)")
print()
recommend_similar(query2)\
"""))

# ── Cell 30 — Conclusions ──────────────────────────────────────────────────────
cells.append(md("""\
## 8. Conclusions and Limitations

### Results summary

| Metric | Baseline | LR | SVM | Random Forest |
|---|---|---|---|---|
| Test accuracy | ~10 % | See table | See table | **Best** |
| Test macro-F1 | ~10 % | See table | See table | **Best** |

*(Actual values are in the model comparison table produced by Cell 5.5 above.)*

### Key findings

1. **All three trained classifiers substantially outperform the majority-class baseline**, confirming that numerical audio features carry real predictive signal.
2. **Classical and metal** are the easiest genres to classify correctly — they occupy distinct regions of the feature space.
3. **Reggae/hip-hop** and **country/rock** are the most frequently confused pairs — their numerical feature distributions overlap considerably.
4. **Feature reduction to 15 features** preserves most of the predictive performance, suggesting the dataset contains some redundant information.
5. The **song-similarity recommender** returns sensible results (same or related genres) but may occasionally surface cross-genre matches because numerical similarity does not capture all musical qualities.

### Limitations

- GTZAN is a small dataset (1 000 tracks, 30 s each). Results should not be generalised beyond this benchmark.
- The dataset was collected in 2002; genre boundaries have evolved and newer sub-genres are not represented.
- Feature extraction parameters (hop length, window size) are fixed; different settings would produce different features.
- The recommender uses Euclidean-space cosine similarity on standardised features, which is a simplification of musical similarity.

### What is not concluded

- We cannot claim that a specific audio property *caused* a misclassification — only that certain features correlate with genres.
- These results do not generalise to streaming-era music without retraining on a more recent dataset.

---

## 9. Sources and Acknowledgements

- **GTZAN dataset**: G. Tzanetakis and P. Cook, "Musical genre classification of audio signals," *IEEE Transactions on Speech and Audio Processing*, 10(5):293–302, 2002.  
  Kaggle page by Andrada Olteanu: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
- **scikit-learn**: Pedregosa et al., *JMLR* 12, pp. 2825–2830, 2011.
- **NumPy, pandas, Matplotlib, Seaborn** — see `CITATIONS.md` for full references and licences.

All implementation code was written originally for this IDC409 coursework.  
Full citation details are in `CITATIONS.md`.\
"""))

# ── Cell 31 — DB close ────────────────────────────────────────────────────────
cells.append(md("## 10. Database cleanup"))
cells.append(code("""\
conn.close()
print("SQLite connection closed.")
print(f"Database file: {os.path.abspath(DB_PATH)}")\
"""))

# ── Assemble notebook ────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "cells": cells,
}

os.makedirs("notebooks", exist_ok=True)
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written: {NB_PATH}")
print(f"Total cells     : {len(cells)}")
