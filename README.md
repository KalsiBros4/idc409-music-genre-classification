# IDC409 — Music Genre Classification

> **Classify songs into music genres using pre-extracted numerical audio features.**  
> Course project for **IDC409** — Data Science and Applications.

---

## Project objective

Given a table of numerical audio features extracted from 30-second song clips (e.g., tempo, spectral centroid, MFCCs), can we automatically assign a genre label?

We use the **GTZAN Music Genre Classification** dataset (1 000 tracks across 10 genres) and compare three classifiers — **Logistic Regression**, **Support Vector Machine (RBF kernel)**, and **Random Forest** — against a majority-class baseline, with feature reduction via SelectKBest.

---

## Course requirements coverage

| Requirement | Implementation |
|---|---|
| NumPy and pandas | Feature matrix, EDA, data cleaning |
| Database (read, write, modify) | SQLite via `src/database_utils.py` and notebook Section 4 |
| Exploratory Data Analysis | Section 3: genre balance, tempo, spectral centroid, MFCC plots, correlation heatmap |
| Feature understanding and reduction | Section 6: SelectKBest k=15 vs full 57 features |
| Machine-learning model | Sections 5–6: Baseline, LR, SVM, Random Forest |
| Evaluation metrics | Accuracy, macro F1, confusion matrix, per-class report |
| GitHub | This repository |

---

## Repository structure

```
idc409-music-genre-classification/
├── README.md                          ← this file
├── requirements.txt                   ← Python dependencies
├── .gitignore
├── CITATIONS.md                       ← full dataset and code attribution
├── build_notebook.py                  ← assembles the .ipynb from cell sources
├── generate_synthetic_data.py         ← creates plausible synthetic data for testing
├── data/
│   └── README.md                      ← download instructions for features_30_sec.csv
├── notebooks/
│   └── music_genre_classification.ipynb  ← MAIN notebook
├── src/
│   └── database_utils.py              ← SQLite helper functions
└── tests/
    └── test_smoke.py                  ← 16 environment + database unit tests
```

---

## Dataset

| Field | Value |
|---|---|
| **Name** | GTZAN Music Genre Classification |
| **Original author** | George Tzanetakis & Perry Cook (2002) |
| **Kaggle URL** | https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification |
| **File used** | `features_30_sec.csv` |
| **Dimensions** | 1 000 rows × 60 columns |
| **Genres** | 10: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock |
| **Features** | 60 raw columns: 57 numerical audio features, `length`, `filename`, and `label`. `length` and `filename` are excluded from model inputs, leaving 57 predictive audio features. |
| **Balance** | Perfectly balanced — 100 tracks per genre |
| **Licence** | CC BY 4.0 (as published on Kaggle) |

The dataset file is **not committed** to this repository (redistribution terms). See `data/README.md` for download instructions.

---

## Setup and installation

### Prerequisites
- Python 3.10 or later
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/KalsiBros4/idc409-music-genre-classification.git
cd idc409-music-genre-classification

# 2. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Obtain the dataset
#    See data/README.md for download instructions.
#    Place  features_30_sec.csv  inside the  data/  folder.

# 5. Launch the notebook
python -m jupyter notebook notebooks/music_genre_classification.ipynb
```

> **Rebuilding & executing the notebook**  
> To rebuild the notebook from source cells and execute it cleanly end-to-end with full outputs:  
> `python build_notebook.py --execute`

> **Quick pipeline test (no real data needed)**  
> Run the synthetic data generator first to verify the full pipeline:  
> `python generate_synthetic_data.py`  
> Then open and run the notebook — all cells execute end-to-end.

---

## Running the tests

```bash
python -m pytest tests/test_smoke.py -v
```

19 focused unit and regression tests verify:
- Environment and library dependencies (NumPy, pandas, scikit-learn, Matplotlib, Seaborn, SQLite3)
- SQLite schema generation, read queries, genre filtering, and UPDATE persistence
- `write_songs()` return value accuracy, ensuring it returns the exact count of newly inserted records and 0 when duplicates are ignored
- Model selection logic, validating that the winning model is selected via mean cross-validation macro F1 on the training set
- GTZAN column schema validation, confirming 60 raw columns yield 57 predictive audio features
- scikit-learn Pipeline construction, DummyClassifier baseline training, and cosine similarity calculation

---

## Database workflow

The notebook (Section 4) demonstrates:

| Operation | Code | What it does |
|---|---|---|
| **Write** | `dbu.write_songs(conn, db_df)` | Inserts all 1 000 songs into SQLite |
| **Read** | `dbu.read_songs_by_genre(conn, "jazz")` | Retrieves rows with a WHERE clause |
| **Modify** | `dbu.update_genre(conn, filename, new_genre)` | Updates a genre field (UPDATE + verify) |

The helper functions are in `src/database_utils.py`. The database file is excluded from Git (it is regenerated by the notebook).

---

## Models and metrics

### Models compared
1. **Baseline** — DummyClassifier (majority class) — lower bound
2. **Logistic Regression** — linear classifier with StandardScaler and L-BFGS solver
3. **SVM (RBF kernel)** — margin-based, C=10, scaled features
4. **Random Forest** — 200 trees, no scaling required

### Evaluation metrics
- Test accuracy
- Test macro F1-score (main metric — treats all genres equally)
- 5-fold stratified cross-validation macro F1 (for model selection)
- Per-class precision, recall, F1 (classification report)
- Confusion matrix (labelled plot)

### Results — Real GTZAN Dataset (verified, executed)

The notebook was executed end-to-end on the real GTZAN dataset (1 000 tracks, 10 genres).

| Model | CV Macro-F1 | Test Accuracy | Test Macro-F1 |
|---|---|---|---|
| Baseline (Majority) | 0.018 | 0.100 | 0.018 |
| Random Forest | 0.697 | 0.690 | 0.682 |
| Logistic Regression | 0.717 | 0.710 | 0.711 |
| **SVM (RBF)** | **0.748** | **0.765** | **0.764** |

**Feature reduction (SelectKBest k=15 vs full 57 features):**

| Model | Full Macro-F1 | Reduced Macro-F1 | Change |
|---|---|---|---|
| Logistic Regression | 0.711 | 0.650 | -0.061 |
| SVM | 0.764 | 0.653 | -0.111 |
| Random Forest | 0.682 | 0.631 | -0.051 |

> On the real dataset, reducing features **decreased** performance — the full 57-feature set carries useful signal that SelectKBest with 15 features discards. This is an honest, informative finding.

**Top 15 features selected (ANOVA F-score):**
`chroma_stft_mean`, `mfcc1_mean`, `spectral_bandwidth_mean`, `rolloff_mean`,
`spectral_centroid_mean`, `spectral_centroid_var`, `mfcc4_mean`, `perceptr_var`,
`mfcc2_mean`, `rms_mean`, `chroma_stft_var` + 4 others

---

## Feature reduction findings

- **Original feature count**: 57 (from 60 raw CSV columns, excluding `length`, `filename`, `label`)
- **Reduced feature count**: 15 (SelectKBest, ANOVA F-score)
- **Top selected features**: include `chroma_stft_mean`, `mfcc1_mean`, `spectral_bandwidth_mean`, `rolloff_mean`, `spectral_centroid_mean`, `spectral_centroid_var`, `mfcc4_mean`, `perceptr_var`, `mfcc2_mean`, `rms_mean`, `chroma_stft_var`.
- **Result**: on the real GTZAN dataset, reducing features from 57 to 15 caused macro-F1 to decline across all three classifiers (LR: -0.061, SVM: -0.111, RF: -0.051). This demonstrates that the full 57-dimensional acoustic feature space carries valuable complementary information that SelectKBest discards.

---

## Genre confusion (audio/AI reflection)

Evidence from the confusion matrix and classification report of the best model (SVM RBF):
- **Metal** achieved the highest precision (0.94) and F1-score (0.86), though 3 tracks were misclassified as blues due to shared guitar distortion and pentatonic blues roots.
- **Classical ↔ Jazz** exhibited the highest mutual confusion (3 classical -> jazz, 2 jazz -> classical; 5 total) due to shared dynamic acoustic instrumentation (pianos, upright basses, woodwinds) and lower continuous percussiveness.
- **Disco** had the lowest recall (0.55, F1 = 0.61), commonly confused with pop (3 tracks), rock (2 tracks), and others due to danceable rhythms, shared instrumentation, and upbeat tempos.
- **Rock ↔ Country** (2 rock -> country, 1 country -> rock) overlap due to similar band arrangements, guitar textures, and mid-tempo rhythmic structures.
- **Pop → Reggae** (3 tracks) and **Pop ↔ Classical** (2 tracks) reflect diverse acoustic arrangements within pop.

---

## Limitations

- GTZAN is small (1 000 clips, 30 s each) and dated (2002). Generalisation to modern streaming music is uncertain.
- Feature extraction parameters are fixed; different windows would change values.
- The song-similarity recommender uses cosine similarity on standardised features — musical similarity in the human sense is not guaranteed.

---

## Possible extensions

- Use Librosa to extract features directly from WAV files.
- Apply a CNN on mel-spectrogram images.
- Tune hyperparameters with GridSearchCV.
- Extend with a modern, larger dataset.

---

## References

See `CITATIONS.md` for full attribution.

- Tzanetakis & Cook (2002). *Musical genre classification of audio signals.* IEEE TSA 10(5).
- scikit-learn: Pedregosa et al., JMLR 12, 2011.
- GTZAN Kaggle page: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
