# imperials-ltd-insurance-analytics
Data Mining project — predicting life insurance purchase likelihood across 25,271 customers using Logistic Regression, KNN &amp; Random Forest in Python. Queen's University Belfast 2024
# Descriptive and Predictive Analytics for Imperials Ltd.

**Module:** MGT7219 – Data Mining  
**Institution:** Queen's University Belfast — MSc Business Analytics  
**Author:** Pradyuman Kumar (Student ID: 40425764)  
**Supervisor:** Dr. V. Charles

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
- [Results](#results)
- [Key Findings](#key-findings)
- [Recommendations](#recommendations)
- [How to Run](#how-to-run)
- [References](#references)

---

## Project Overview

This project implements supervised machine learning to help **Imperials Ltd.**, an insurance company, identify and rank prospective life insurance purchasers from their existing customer base.

The goal is to build predictive models that can:
- Detect patterns in customer behaviour
- Anticipate purchase inclinations
- Improve targeting of marketing campaigns
- Drive long-term consumer loyalty

Three classification algorithms are compared:

| Algorithm | Accuracy | AUC Score | F1-Score |
|---|---|---|---|
| ✅ Logistic Regression | 68.99% | 0.747 | 73.54% |
| KNN Classifier | 64.05% | 0.672 | 68.82% |
| Random Forest | 65.25% | — | 69.88% |

**Winner: Logistic Regression** — best overall accuracy and F1-score.

---

## Dataset Description

The dataset is sourced from the customer database of Imperials Ltd. and contains **25,271 observations across 13 features.**

| Feature | Type | Description |
|---|---|---|
| flag | Binary (Y/N) | Target variable — did the customer respond to the campaign? |
| gender | Categorical | Customer gender (M/F) |
| education | Categorical | Education level (0.<HS to 4.Grad) |
| house_val | Continuous | Estimated home value |
| age | Categorical | Age group (<=25 to >65) |
| online | Binary (Y/N) | Online presence |
| marriage | Categorical | Marital status |
| child | Categorical | Has children (Y/N) |
| occupation | Categorical | Job type (Professional, Blue Collar, etc.) |
| mortgage | Categorical | Mortgage level (Low/Med/High) |
| house_owner | Binary | Owner or Renter |
| region | Categorical | Geographic region (South, West, etc.) |
| fam_income | Ordinal | Family income band |

**Key statistics:**
- 51.59% of customers responded positively to the marketing campaign
- 75.87% of customers are homeowners
- 57.47% are married; 66.15% have children
- 58.53% work in white-collar or professional roles

*Note: The dataset was provided as part of the MSc programme at Queen's University Belfast and is not included in this repository. The code and analysis are original work.*

---

## Methodology

### Data Cleaning

Five key cleaning steps were applied:

| Step | Column | Action |
|---|---|---|
| 1 | child | Replaced 0 values with N (no child) |
| 2 | marriage | Inferred missing marital status using age and child data |
| 3 | education | Dropped 741 rows with missing values |
| 4 | gender | Removed unknown U values |
| 5 | house_val | Imputed missing values using regional mean; binned into 20 categories |

### Data Conversion

All categorical variables were encoded to numeric form for ML compatibility:

```python
# Example encodings used
flag:        Y → 1,  N → 0
gender:      M → 1,  F → 0
marriage:    Married → 1,  Single → 0
education:   0.<HS → 0,  1.HS → 1,  2.Some College → 2,  3.Bach → 3,  4.Grad → 4
```

One-hot encoding (`pd.get_dummies`) was applied to all multi-class categorical variables. Features were scaled using `StandardScaler`.

### Exploratory Data Analysis

Five visualisations were produced to understand customer profiles:
- Distribution of Occupation — Professionals dominate; farm workers least represented
- Purchase Behaviour by Education — Bachelor's-level customers buy most
- Purchase Behaviour by Gender — Males purchase more than females
- Occupation & Gender of Buyers — Professional males are the top buyers
- Distribution of Regions — South leads with 39.2%, followed by West (21.7%)

### Model Building

Dataset split: **70% training / 30% testing** using `train_test_split` with `random_state=42`

**Model 1 — Logistic Regression**
```python
model = LogisticRegression()
model.fit(X_train, y_train)
```

**Model 2 — KNN Classifier**
```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
```

Performance assessed using: Accuracy Score, ROC-AUC, F1-Score and Confusion Matrix.

---

## Results

### Logistic Regression — Confusion Matrix

| | Predicted: No (0) | Predicted: Yes (1) |
|---|---|---|
| **Actual: No (0)** | 1965 (TN) | 1372 (FP) |
| **Actual: Yes (1)** | 979 (FN) | 3266 (TP) |

- Accuracy: **68.99%**
- AUC: **0.747**
- F1-Score: **73.54%**

### KNN Classifier — Confusion Matrix

| | Predicted: No (0) | Predicted: Yes (1) |
|---|---|---|
| **Actual: No (0)** | 1800 (TN) | 1500 (FP) |
| **Actual: Yes (1)** | 1200 (FN) | 3000 (TP) |

- Accuracy: **64.05%**
- AUC: **0.672**
- F1-Score: **68.82%**

---

## Key Findings

- Logistic Regression outperforms KNN on all metrics and is the recommended production model
- The South region is the strongest market (39.2% of customers)
- Professional males with at least a bachelor's degree are the highest-converting customer segment
- 51.59% of existing customers responded positively to the previous campaign — a strong baseline
- Customers with children are highly likely to be married, which can guide data imputation in future datasets

---

## Recommendations

- Deploy Logistic Regression as the primary model for scoring leads in future marketing campaigns
- Conduct ongoing model monitoring — retrain periodically as new customer data is collected
- Investigate the female segment — understand barriers to purchase and create tailored campaigns
- Expand in the South — already the largest region; increased investment could yield higher ROI
- Explore ensemble methods (XGBoost, Gradient Boosting) in future iterations for potentially higher accuracy

---

## How to Run

### Prerequisites
Python 3.8+ required. Developed in Google Colab but can be run locally.

### Installation

```bash
git clone https://github.com/Pradyuman291/imperials-ltd-insurance-analytics.git
cd imperials-ltd-insurance-analytics
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Running the Code

**Option A — Google Colab (Recommended)**
1. Open Google Colab
2. Upload `imperials_analytics.ipynb`
3. Upload `sales_data.csv` when prompted
4. Run all cells (Runtime > Run all)

**Option B — Local Jupyter Notebook**
```bash
pip install jupyter
jupyter notebook notebooks/imperials_analytics.ipynb
```

**Option C — Python Script**
```bash
python src/main.py
```

---

## Technologies Used

- **Language:** Python
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- **Environment:** Google Colab / Jupyter Notebook

---

## References

- Boodhun, N. and Jayabalan, M. (2018). Risk prediction in life insurance industry using supervised learning algorithms. *Complex & Intelligent Systems*, 4(2), 145–154.
- Kurt, Z. et al. (2022). Insurance sales forecast using machine learning algorithms. *Lecture Notes in Networks and Systems*, 29–38.
- Hastie, T., Tibshirani, R. and Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- Brownlee, J. (2016). *Machine Learning Mastery with Python*. Machine Learning Mastery.

---

*This project was submitted as academic coursework at Queen's University Belfast (MGT7219 Data Mining). The code is shared for educational reference only.*

Made with 🐍 Python & 📊 scikit-learn | Queen's University Belfast
