# Customer Spending Intelligence

A end-to-end fintech data science project built on 1.3 million real credit card transactions. The project covers the full pipeline from raw data exploration through machine learning modeling to a deployed interactive application — the kind of work a data scientist or analyst would be expected to produce in a fintech or banking context.

**Live app:** [huggingface.co/spaces/ShreySheth91/customer-spending-intelligence](https://huggingface.co/spaces/ShreySheth91/customer-spending-intelligence)

---

## What this project does

Most portfolio projects stop at a notebook. This one goes further — the EDA findings feed into the clustering model, the clustering output feeds into the fraud model as a feature, and the fraud model is served through a live app where you can score transactions in real time and get a SHAP explanation for each prediction.

The dataset is a Kaggle credit card transactions dataset with 1.3M rows, 23 columns, and a 0.58% fraud rate — a realistic class imbalance that forces you to think carefully about evaluation metrics and modeling strategy rather than just fitting a classifier and reporting accuracy.

---

## Project structure

```
customer-spending-intelligence/
│
├── 01_eda_customer_spending.ipynb     Phase 1 — Exploratory data analysis
├── 02_rfm_segmentation.ipynb          Phase 2 — RFM segmentation and K-Means clustering
├── 03_fraud_modeling.ipynb            Phase 3 — Fraud detection and CLV regression
│
├── customer_aggregated.csv            Customer-level feature table (output of Phase 1)
├── customer_segments.csv              Cluster labels appended (output of Phase 2)
├── fraud_model.json                   Trained XGBoost fraud classifier
├── clv_model.json                     Trained XGBoost CLV regressor
├── model_metadata.json                Feature lists, thresholds, and metric scores
│
├── app.py                             Streamlit app (4 pages)
├── requirements.txt                   Python dependencies
├── dashboard.html                     Standalone EDA dashboard (no server needed)
│
├── plots/                             All saved figures from Phases 1-3
└── README.md
```

> The raw dataset (credit_card_transactions.csv, ~346 MB) is not committed to this repo. Download it from [Kaggle](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset) and place it in the project root before running the notebooks.

---

## Phase 1 — Exploratory Data Analysis

The EDA notebook covers the full transaction landscape across 983 customer accounts. Key findings:

The fraud rate is 0.58% — 7,506 fraudulent transactions out of 1,296,675. This level of class imbalance makes accuracy a meaningless metric. A model that predicts "not fraud" for every transaction scores 99.4% accuracy while catching zero fraud. Precision-Recall AUC is the correct evaluation metric here.

Fraudulent transactions carry a median amount of $396.50 versus $47.28 for legitimate ones — an 8.4x premium. Online merchant categories (shopping_net, misc_net) carry fraud rates 2-3x above the dataset average, while physical point-of-sale categories like health_fitness and home are the safest. Fraud clusters sharply in late-night hours (12am-4am) when cardholder transaction volume is at its lowest.

Geographic distance between cardholder and merchant turns out to be a weak standalone signal — the median distance is nearly identical for fraudulent (77.9 km) and legitimate (78.2 km) transactions. Time of day and transaction amount relative to the customer's own spending norm are far more predictive.

The notebook builds a customer-level aggregation table with RFM features (recency, frequency, monetary), behavioral features (night transaction percentage, spend volatility, merchant diversity), and a Haversine distance calculation — 983 rows, 22 features, saved as customer_aggregated.csv for Phase 2.

---

## Phase 2 — RFM Segmentation

The segmentation notebook takes the customer-level table and clusters it into four behavioral segments using K-Means. The optimal cluster count is selected via Elbow curve and Silhouette score analysis rather than picking k arbitrarily. Clusters are validated visually with PCA and t-SNE projections and profiled with radar charts.

The four segments that emerged from the data:

**Champions** (396 customers, 40.3%) — most recent activity, highest frequency (2,062 transactions average), highest lifetime spend ($143K average). These are the highest-value accounts in the portfolio.

**Loyal** (512 customers, 52.1%) — recent, high-frequency, solid spend ($66K average). The core revenue base and the largest segment by count.

**Suspicious** (34 customers, 3.5%) — dormant for 250+ days, very low frequency but anomalously high average transaction ($640, nearly 10x the dataset average). Weekend-heavy (79% of transactions on weekends). 100% fraud-flagged. Large infrequent transactions on inactive accounts concentrated on weekends is a textbook fraud pattern.

**Dormant** (41 customers, 4.2%) — inactive for ~300 days, minimal transactions, low spend, older demographic. 100% fraud-flagged.

The cluster label is exported as a feature in customer_segments.csv and carried into Phase 3 where it gives the fraud model a segment-level signal on top of the transaction-level features.

---

## Phase 3 — Fraud Detection and CLV Modeling

### Fraud classifier

The fraud model is trained at transaction level (1.3M rows) with XGBoost. Key engineering decisions:

**SMOTE at 10:1 ratio** rather than full balancing. Oversampling to 1:1 hurt precision on this dataset because synthetic fraud samples started overlapping with legitimate transaction space. A 10:1 legit-to-fraud ratio preserves the signal while giving the model more fraud examples to learn from than the raw 172:1.

**Cyclical hour encoding** — hour of day is encoded as sin and cos components rather than a raw integer. This ensures the model understands that 11pm and 12am are adjacent, not maximally different. A raw integer encoding would make midnight look like the furthest point from itself.

**Customer-normalized amount** — amt_z_customer measures how many standard deviations a transaction's amount is from that specific customer's own historical mean. A $900 transaction from a customer who usually spends $50 is very different from a $900 transaction from a high-spending customer. This feature captures that context that a raw amount cannot.

**Hyperparameter tuning** via RandomizedSearchCV with 30 iterations and 3-fold stratified cross-validation, scoring on average_precision (PR-AUC). The optimal decision threshold is found by maximizing F1 score across the precision-recall curve rather than defaulting to 0.5 — which on this imbalanced dataset would produce a model that almost never flags fraud.

SHAP values are computed on a 5,000-sample subset for the beeswarm, waterfall, and dependence plots. The waterfall plot explains any individual prediction — which features pushed the score up, which pulled it down, and by how much.

### CLV regression

Customer lifetime value is modeled as a regression problem at the customer level (983 rows). The target is log-transformed total spend to reduce right skew. The model achieves R² of 0.997 and MAE of $1,586 on the test set.

SHAP attribution shows frequency is the dominant driver (mean absolute SHAP value 0.47), followed by unique_merchants (0.19), avg_txn_amt (0.15), and recency (0.14). The remaining features contribute essentially nothing — CLV is almost entirely determined by how often a customer transacts, how broadly they shop, and how recently they were active. This is consistent with the theoretical RFM framework and validates that the feature engineering in Phase 1 was on the right track.

---

## Live application

The Streamlit app has four pages:

**EDA Overview** — key metrics, fraud distribution charts, category spend vs. fraud rate, and the hourly transaction heatmap showing the 12am-4am fraud spike.

**Fraud Scorer** — input a transaction (amount, hour, day, category, distance, customer segment) and get a fraud probability score with a SHAP waterfall chart explaining which features drove the prediction up or down. The model flags transactions above the optimal F1 threshold computed during training.

**Segment Explorer** — browse all four customer segments with radar profiles, a detail card with business recommendation for each, and feature distribution histograms loaded from the actual customer data.

**CLV Estimator** — input a customer profile and get a predicted lifetime value with SHAP attribution showing which behavioral features most influenced the estimate.

---

## Results summary

| Model | Metric | Score |
|---|---|---|
| Fraud classifier | PR-AUC | see model_metadata.json |
| Fraud classifier | ROC-AUC | see model_metadata.json |
| CLV regressor | R² | 0.997 |
| CLV regressor | MAE | $1,586 |

---

## Modeling decisions worth noting in interviews

| Decision | Why |
|---|---|
| PR-AUC over accuracy | 99.4% accuracy = zero fraud caught on this dataset |
| SMOTE at 10:1 not 1:1 | Full balancing collapses precision; 10:1 preserves the signal |
| Cyclical hour encoding | sin/cos prevents the model treating 11pm and 12am as maximally different |
| amt_z_customer | Contextualises spend relative to that customer's own norm, not the population |
| Threshold at max-F1, not 0.5 | Default 0.5 threshold catches almost no fraud on an imbalanced dataset |
| SHAP over gain importance | Gain shows which features are used; SHAP shows direction and magnitude per prediction |
| Cluster label as fraud feature | Segment identity (Suspicious, Dormant) carries predictive signal beyond individual transaction features |

---

## Stack

Python 3.10 — pandas, numpy, scikit-learn, imbalanced-learn, XGBoost, SHAP, matplotlib, seaborn, plotly, Streamlit. Notebooks designed to run locally or on Google Colab Pro. App deployed on Hugging Face Spaces.

---

## Running locally

```bash
git clone https://github.com/ShreySheth91/customer-spending-intelligence.git
cd customer-spending-intelligence

pip install -r requirements.txt

# Download the dataset from Kaggle and place as:
# credit_card_transactions.csv

# Run notebooks in order
jupyter notebook 01_eda_customer_spending.ipynb

# Launch the app
streamlit run app.py
```

---

## Dataset

[Credit Card Transactions Dataset — priyamchoksi (Kaggle)](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset)

1,296,675 transactions across 983 customers, January 2019 to June 2020, 23 columns including transaction timestamp, amount, merchant category, cardholder demographics, coordinates, and a binary fraud label.

---

Shrey Sheth — MS Computer Science (Data Science & AI), Pace University  
[GitHub](https://github.com/ShreySheth91)
[LinkedIn](https://www.linkedin.com/in/shreysheth91/)
