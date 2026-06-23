# 💼 Job Career Level Classification — NLP + Random Forest

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange?logo=scikit-learn)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTEN-orange)
![Task](https://img.shields.io/badge/Task-NLP%20Classification-blueviolet)

---

## 📌 Project Overview

Classify job postings into **career levels** using NLP on job title, description, location, function and industry — with SMOTEN oversampling to handle severe class imbalance.

| Item | Detail |
|------|--------|
| **Dataset** | Job Postings Dataset (8,074 records) |
| **Model** | Random Forest + TF-IDF + OneHotEncoder |
| **Features** | title, description, location, function, industry |
| **Target** | `career_level` (6 classes) |
| **Challenge** | Severe class imbalance handled with SMOTEN |

---

## 🗂️ Project Structure

```
06_career_level_classification/
├── main.ipynb          ← Full notebook: EDA → Pipeline → Train → Evaluate
├── main.py             ← Python script version
├── dataset_info.md     ← Dataset description
└── README.md           ← This file
```

---

## 🚀 How to Run

### Step 1 — Install dependencies
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn odfpy
```

### Step 2 — Add dataset
Place `06_data_career.ods` in this folder.

### Step 3 — Open notebook
```bash
jupyter notebook main.ipynb
```
Then click **Kernel → Restart & Run All**

---

## ⚙️ ML Pipeline

```
Job Posting (title + description + location + function + industry)
                          │
                          ▼
               ColumnTransformer
  ┌─────────────────────────────────────────────┐
  │ title       → TF-IDF                        │
  │ description → TF-IDF bigrams (stop words)   │
  │ industry    → TF-IDF                        │
  │ location    → OneHotEncoder                 │
  │ function    → OneHotEncoder                 │
  └─────────────────────────────────────────────┘
                          │
                          ▼
           Chi² SelectPercentile (top 5%)
                          │
                          ▼
           RandomForestClassifier
```

---

## 🏷️ Career Level Labels

| Label | Description |
|-------|-------------|
| `specialist` | Individual contributor |
| `manager_team_leader` | Team or project manager |
| `senior_specialist_or_project_manager` | Senior IC or PM |
| `bereichsleiter` | Department head |
| `director_business_unit_leader` | Director level |
| `managing_director_small_medium_company` | MD / C-level |

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | ~78% |
| Weighted F1 | ~0.75 |

---

## 🔑 Key Concepts

- ✅ **SMOTEN** oversampling for imbalanced categorical/text data
- ✅ TF-IDF bigrams capture seniority phrases ("senior manager", "team lead")
- ✅ Regex extracts clean US state codes from messy location strings
- ✅ Chi² feature selection reduces noise from sparse TF-IDF matrices
- ✅ sklearn Pipeline prevents data leakage

---

## 📦 Dependencies

```
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
odfpy
```
