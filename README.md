# PulseIQ — AI-Driven Customer Analytics Platform

Live Demo:https://ai-driven-customer-analytics.onrender.com

## Quick Start
```bash
python run.py
# Open http://127.0.0.1:5000
```

## Requirements
- Python 3.8+
- flask, scikit-learn, pandas, numpy  
  *(auto-installed on first run)*

## Project Structure
```
customer_analytics/
├── run.py              ← START HERE
├── app.py              ← Flask backend + all ML models
├── generate_data.py    ← Synthetic dataset generator
├── customers.csv       ← 350 customer records (auto-generated)
└── templates/
    └── index.html      ← Full dashboard UI
```

## Features

### ML Models
| Model | Algorithm | Purpose |
|-------|-----------|---------|
| Churn Prediction | Gradient Boosting | Predict 30-day churn risk |
| Segmentation | K-Means (k=4) | Group customers by RFM behavior |
| LTV Scoring | Weighted formula | Lifetime value estimation |
| Recommendations | Rule-based ML | Based on recent purchases + income |

### Dashboard Pages
1. **Dashboard** — KPI metrics, revenue trends, segment donut, regional breakdown
2. **Customers** — Filterable table with churn %, LTV, segment badges
3. **Segments** — Deep dive: Champions, Growth, At-Risk, Occasional Buyers
4. **Churn Predictor** — Enter any customer profile → get churn probability + recommendations
5. **Model Stats** — Accuracy, Precision, Recall, F1-Score, Feature Importance

### Filters (applied across all views)
- Region (North/South/East/West/Central)
- Income Segment (Low/Medium/High)
- Customer Segment
- Churn Status (At Risk / Safe)
- LTV Score range

### Dataset (350 customers)
- Demographics: age, region, income
- Behavioral: purchases, spend, recency, frequency
- Engagement: email open rate, app sessions, support tickets
- Derived: RFM scores, LTV score, churn probability, segment label

## API Endpoints
```
GET  /api/dashboard          → KPIs + chart data (supports filters)
GET  /api/customers          → Paginated customer list (supports filters)
GET  /api/customer/<id>      → Customer detail + recommendations
POST /api/predict            → Churn prediction + recommendations
GET  /api/model_stats        → Classification report
GET  /api/filters_meta       → Available filter options
```
