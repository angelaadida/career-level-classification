# 📊 Dataset Info — Job Postings Career Level Dataset

## 📋 About This Dataset

This is the course dataset `06_data_career.ods` — a collection of real job postings labeled with career levels.

---

## 📋 Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| `title` | str | Job title (e.g. "Senior Software Engineer") |
| `location` | str | Job location (e.g. "New York, NY") |
| `description` | str | Full job description text |
| `function` | str | Job function category |
| `industry` | str | Industry sector |
| `career_level` | str | **Target** — Career level label |

---

## 🏷️ Career Level Labels & Distribution

| Label | Count |
|-------|-------|
| `senior_specialist_or_project_manager` | 4,338 |
| `manager_team_leader` | 2,672 |
| `bereichsleiter` | 960 |
| `director_business_unit_leader` | 70 |
| `specialist` | 30 |
| `managing_director_small_medium_company` | 4 |
| **Total** | **8,074** |

> ⚠️ **Severe class imbalance!** — handled with **SMOTEN** oversampling in the pipeline.

---

## 🔧 How to Use

1. Place `06_data_career.ods` in the same folder as `main.ipynb`
2. Run `main.ipynb` — the notebook reads it automatically

---

## 💡 Notes

- Location strings are cleaned with regex to extract US state codes
- `odfpy` library required to read `.ods` files: `pip install odfpy`
