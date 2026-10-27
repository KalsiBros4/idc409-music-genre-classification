# Data Directory

## What goes here

Place the GTZAN dataset CSV file here before running the notebook.

## Required file

`features_30_sec.csv`

## How to download

1. Go to the Kaggle dataset page:  
   https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

2. Sign in or create a free Kaggle account.

3. Click **Download** (top-right of the dataset page). You will receive a zip archive.

4. Extract the archive. Inside, locate:  
   `Data/features_30_sec.csv`

5. Copy **`features_30_sec.csv`** into this `data/` folder.

## File description

| Property | Value |
|---|---|
| Source | GTZAN Music Genre Classification (Kaggle / George Tzanetakis, 2002) |
| File | `features_30_sec.csv` |
| Rows | 1,000 (one per 30-second audio clip) |
| Columns | 60 (58 audio features + `filename` + `label`) |
| Classes | 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock |
| Missing values | None expected |
| Licence | The GTZAN dataset is widely used for academic research. The Kaggle page republishes it under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Verify licence terms before redistribution. |

## Why this file is not committed

Large dataset files are excluded from the repository to respect the dataset's redistribution terms and to keep the repository lightweight. The notebook recreates all results when the file is placed here.
