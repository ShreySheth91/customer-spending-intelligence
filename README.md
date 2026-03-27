# Customer Spending Intelligence
### Fintech Portfolio Project — Phase 1: Exploratory Data Analysis

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?style=flat&logo=pandas)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Phase%201%20Complete-52d48b?style=flat)

> **A full-stack data science project on 1.3M credit card transactions** — covering EDA, RFM segmentation, fraud detection, and customer lifetime value modeling.

---

## Dashboard

**[→ View Live EDA Dashboard](https://htmlpreview.github.io/?https://github.com/ShreySheth91/customer-spending-intelligence/blob/main/dashboard.html)**

The interactive dashboard covers all Phase 1 findings: fraud distribution, category-level risk, hourly heatmap, demographic breakdown, and engineered RFM features.

---

## Project Structure

```
customer-spending-intelligence/
│
├── 01_eda_customer_spending.ipynb   # Phase 1 — EDA notebook (1.3M rows)
├── customer_aggregated.csv          # Customer-level RFM feature table (983 rows × 22 cols)
├── dashboard.html                   # Standalone interactive EDA dashboard
├── plots/                           # All saved matplotlib / seaborn figures
│   ├── 01_fraud_distribution.png
│   ├── 02_amount_distribution.png
│   ├── 03_category_analysis.png
│   ├── 04_hourly_pattern.png
│   ├── 05_daily_pattern.png
│   ├── 06_monthly_trends.png
│   ├── 07_age_analysis.png
│   ├── 08_job_spend.png
│   ├── 09_distance_analysis.png
│   ├── 10_customer_features.png
│   ├── 11_correlation_heatmap.png
│   └── 12_fraud_correlation.png
└── README.md
```

**Note:** `credit_card_transactions.csv` (~346 MB) is excluded from the repo. Download it from [Kaggle — priyamchoksi/credit-card-transactions-dataset](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset) and place it in the project root.

---

## Dataset

| Attribute | Value |
|---|---|
| Source | Kaggle — priyamchoksi |
| Rows | 1,296,675 |
| Columns | 23 |
| Date range | Jan 2019 – Jun 2020 |
| Target | `is_fraud` (binary) |
| Fraud rate | **0.58%** — severe class imbalance |

---

## Key Findings

### Fraud Landscape
- Only **0.58%** of transactions are fraudulent (7,506 of 1,296,675)
- Fraudulent transactions carry a **median amount of $396.50** vs $47.28 for legitimate — an **8.4× premium**
- **77.5%** of customer accounts have at least one fraudulent transaction in history

### Category Risk
| Category | Fraud Rate | Risk |
|---|---|---|
| shopping_net | 1.76% | 🔴 High |
| misc_net | 1.45% | 🔴 High |
| grocery_pos | 1.41% | 🔴 High |
| health_fitness | 0.15% | 🟢 Low |
| home | 0.16% | 🟢 Low |

Online categories carry 2–3× the fraud rate of physical POS channels.

### Temporal Signals
- **Peak fraud hours: 12am–4am** — when cardholder volume is lowest
- **Weekends** show elevated fraud rates vs weekday average
- Engineered features: `trans_hour`, `is_weekend`, `night_txn_pct`

### Geographic Distance
- Median distance: Legit **78.2 km** vs Fraud **77.9 km** — distance alone is a **weak signal**
- Combine with hour + amount for compound fraud features

---

## Feature Engineering

Customer-level RFM table built from transaction history (`customer_aggregated.csv`):

| Feature | Type | Description |
|---|---|---|
| `recency` | RFM | Days since last transaction |
| `frequency` | RFM | Total transaction count |
| `monetary` | RFM | Total lifetime spend |
| `night_txn_pct` | Fraud | % of transactions between 12am–6am |
| `max_txn_amt` | Fraud | Largest single transaction |
| `std_txn_amt` | Behavioral | Spending volatility |
| `avg_distance_km` | Behavioral | Haversine cardholder → merchant |
| `unique_merchants` | Diversity | Merchant breadth |
| `unique_states` | Diversity | Geographic spread |

---

## Project Roadmap

- [x] **Phase 1** — EDA & Feature Engineering *(this repo)*
- [ ] **Phase 2** — RFM Segmentation (K-Means, Silhouette scoring)
- [ ] **Phase 3** — Fraud / Churn Modeling (XGBoost + SMOTE, PR-AUC, SHAP)
- [ ] **Phase 4** — Interactive App (Streamlit / Hugging Face Spaces)

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/ShreySheth91/customer-spending-intelligence.git
cd customer-spending-intelligence

# Install dependencies
pip install pandas numpy matplotlib seaborn plotly scikit-learn

# Download the dataset from Kaggle and place it as:
# credit_card_transactions.csv

# Launch the notebook
jupyter notebook 01_eda_customer_spending.ipynb
```

---

## Author

**Shrey Sheth** — MS Computer Science (Data Science & AI), Pace University  
Technical Lead @ Google Developer Group on Campus  
[GitHub](https://github.com/ShreySheth91) · [LinkedIn](https://linkedin.com/in/shreysheth)
