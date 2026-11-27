# Citations and Attributions

This file records all datasets, documentation, papers, and third-party code consulted or used in this project.

---

## 1. Dataset

### GTZAN Music Genre Classification

| Field | Detail |
|---|---|
| **Title** | GTZAN Music Genre Classification |
| **Original author** | George Tzanetakis and Perry Cook (2002) |
| **Kaggle publisher** | Andrada Olteanu |
| **Kaggle URL** | https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification |
| **Original paper** | G. Tzanetakis and P. Cook, "Musical genre classification of audio signals," *IEEE Transactions on Speech and Audio Processing*, vol. 10, no. 5, pp. 293–302, July 2002. DOI: 10.1109/TSA.2002.800560 |
| **Licence** | CC BY 4.0 (as published on Kaggle) |
| **Usage in project** | The file `features_30_sec.csv` is used as the primary dataset. Not redistributed; users download it directly from Kaggle. |

### GACMIS (Indian Music Genres)

| Field | Detail |
|---|---|
| **Title** | GACMIS — Genre Automated Classification of Indian Songs |
| **URL** | https://github.com/ujjwalll/GACMIS |
| **Usage in project** | Optional extended dataset providing 598 `.opus` tracks across 7 Indian/Hindi genres (`bollywood_rap`, `bollywood_romantic`, `sufi`, `ghazal`, `bhajan`, `bhojpuri`, `garhwali`). |

### Navrasa 5000 (`India/Punjabi_Pop`)

| Field | Detail |
|---|---|
| **Title** | Navrasa 5000 Music Dataset (`beastLucifer/navrasa-5000-dataset`) |
| **URL** | https://huggingface.co/datasets/beastLucifer/navrasa-5000-dataset |
| **Licence** | CC0-1.0 |
| **Usage in project** | Optional extended dataset providing 100 `.wav` clips from `India/Punjabi_Pop.zip` for `punjabi` genre classification. |

---

## 2. Python Libraries

All libraries are open-source and used under their respective licences.

| Library | Version (min) | Licence | URL |
|---|---|---|---|
| NumPy | 1.24 | BSD-3-Clause | https://numpy.org |
| pandas | 2.0 | BSD-3-Clause | https://pandas.pydata.org |
| scikit-learn | 1.3 | BSD-3-Clause | https://scikit-learn.org |
| Matplotlib | 3.7 | PSF (BSD-compatible) | https://matplotlib.org |
| Seaborn | 0.12 | BSD-3-Clause | https://seaborn.pydata.org |
| Jupyter | 7.0 | BSD-3-Clause | https://jupyter.org |

---

## 3. Documentation Consulted

- NumPy documentation — https://numpy.org/doc/stable/
- pandas documentation — https://pandas.pydata.org/docs/
- scikit-learn user guide — https://scikit-learn.org/stable/user_guide.html
- Matplotlib documentation — https://matplotlib.org/stable/contents.html
- Seaborn tutorial — https://seaborn.pydata.org/tutorial.html
- Python sqlite3 module — https://docs.python.org/3/library/sqlite3.html
- scikit-learn Pipeline — https://scikit-learn.org/stable/modules/compose.html
- SelectKBest feature selection — https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html

---

## 4. Reference Papers and Articles

- G. Tzanetakis and P. Cook, "Musical genre classification of audio signals," *IEEE Transactions on Speech and Audio Processing*, vol. 10, no. 5, pp. 293–302, 2002.  
  Foundational work that defined the GTZAN benchmark.

---

## 5. Code Attribution

All implementation code in this project was written originally for this IDC409 coursework.  
No code was directly copied from any third-party repository.

General ideas and technique references:
- scikit-learn example notebooks (Apache 2.0 licence): https://scikit-learn.org/stable/auto_examples/
- sklearn Pipeline usage patterns inspired by official scikit-learn documentation.
- Cosine-similarity recommender pattern inspired by sklearn `cosine_similarity` documentation.

---

## 6. Academic Integrity Statement

This project was completed for IDC409 coursework. All analysis, implementation, and conclusions are the student's own work. Third-party sources are cited above. The dataset was obtained through its public Kaggle page and is not redistributed here.
