from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
df_raw = None
models = {}
scalers = {}
encoders = {}
cluster_labels = None
churn_accuracy = None
churn_report = None

CATEGORY_RECS = {
    "Electronics":    ["Laptop Stand","Wireless Earbuds","Smart Watch","Phone Case","USB Hub"],
    "Clothing":       ["Premium T-Shirt","Casual Jeans","Running Shoes","Winter Jacket","Cotton Kurta"],
    "Groceries":      ["Organic Dry Fruits","Healthy Snack Box","Protein Powder","Green Tea Pack","Millet Pack"],
    "Beauty":         ["Vitamin C Serum","Moisturizer Kit","Hair Oil","Sunscreen SPF50","Face Mask Set"],
    "Sports":         ["Yoga Mat","Resistance Bands","Water Bottle","Gym Gloves","Jump Rope"],
    "Home & Kitchen": ["Air Fryer","Coffee Maker","Non-stick Pan Set","Smart LED Bulbs","Storage Organizer"],
    "Books":          ["Self-Help Bestseller","Fiction Novel","Business Strategy","Python Programming","Cookbook"],
    "Toys":           ["STEM Building Kit","Board Game","Educational Puzzle","Remote Car","Art Set"]
}

SEGMENT_NAMES = {0: "Champions", 1: "Growth Potential", 2: "At-Risk", 3: "Occasional Buyers"}
SEGMENT_COLORS = {0: "#185FA5", 1: "#3B6D11", 2: "#993C1D", 3: "#888780"}

def load_and_train():
    global df_raw, models, scalers, encoders, cluster_labels, churn_accuracy, churn_report

    df_raw = pd.read_csv("customers.csv")

    # ── Encode income_segment ─────────────────────────────────────────────────
    le = LabelEncoder()
    df_raw["income_segment_enc"] = le.fit_transform(df_raw["income_segment"])
    encoders["income_segment"] = le

    # ── CHURN PREDICTION ──────────────────────────────────────────────────────
    churn_features = ["age","recency_score","frequency_score","monetary_score",
                      "support_tickets","email_open_rate","app_sessions_monthly",
                      "loyalty_years","avg_order_value","income_segment_enc"]

    X = df_raw[churn_features]
    y = df_raw["churned"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    clf = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    churn_accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    churn_report = classification_report(y_test, y_pred, output_dict=True)
    models["churn"] = clf

    # predict churn probability for all customers
    df_raw["predicted_churn_prob"] = clf.predict_proba(X)[:, 1].round(3)
    df_raw["predicted_churned"] = clf.predict(X)

    # ── K-MEANS SEGMENTATION ──────────────────────────────────────────────────
    seg_features = ["recency_score","frequency_score","monetary_score","ltv_score","income_segment_enc"]
    scaler = StandardScaler()
    X_seg = scaler.fit_transform(df_raw[seg_features])
    scalers["kmeans"] = scaler

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_raw["segment"] = kmeans.fit_predict(X_seg)
    models["kmeans"] = kmeans

    # Reassign segment labels by mean LTV (0=highest)
    seg_ltv = df_raw.groupby("segment")["ltv_score"].mean().sort_values(ascending=False)
    seg_map = {old: new for new, old in enumerate(seg_ltv.index)}
    df_raw["segment"] = df_raw["segment"].map(seg_map)
    df_raw["segment_name"] = df_raw["segment"].map(SEGMENT_NAMES)

    print(f"✓ Dataset: {len(df_raw)} customers loaded")
    print(f"✓ Churn model accuracy: {churn_accuracy}%")
    print(f"✓ Segments: {df_raw['segment_name'].value_counts().to_dict()}")

# ── HELPERS ────────────────────────────────────────────────────────────────────
def get_recommendations(customer_row):
    recent = json.loads(customer_row["recent_purchases"])
    preferred = customer_row["preferred_category"]
    income = customer_row["income_segment"]
    ltv = customer_row["ltv_score"]

    recs = []
    # Primary: recent categories
    for cat in recent[:2]:
        items = CATEGORY_RECS.get(cat, [])
        if items:
            recs.append({"category": cat, "product": np.random.choice(items),
                         "reason": f"Based on recent {cat} purchase", "confidence": round(0.85 + ltv/1000*0.1, 2)})
    # Secondary: preferred
    if preferred not in recent:
        items = CATEGORY_RECS.get(preferred, [])
        if items:
            recs.append({"category": preferred, "product": np.random.choice(items),
                         "reason": f"Your favourite category", "confidence": round(0.78, 2)})
    # Income-based
    if income == "High" and "Electronics" not in recent:
        recs.append({"category": "Electronics", "product": np.random.choice(CATEGORY_RECS["Electronics"]),
                     "reason": "Premium product match for your profile", "confidence": 0.72})
    elif income == "Low" and "Groceries" not in recent:
        recs.append({"category": "Groceries", "product": np.random.choice(CATEGORY_RECS["Groceries"]),
                     "reason": "Best value for your budget", "confidence": 0.68})
    return recs[:4]

def apply_filters(df, params):
    filtered = df.copy()
    region = params.get("region", "All")
    income_seg = params.get("income_segment", "All")
    seg_name = params.get("segment", "All")
    churn_filter = params.get("churn", "All")
    min_ltv = float(params.get("min_ltv", 0))
    max_ltv = float(params.get("max_ltv", 100))

    if region != "All":
        filtered = filtered[filtered["region"] == region]
    if income_seg != "All":
        filtered = filtered[filtered["income_segment"] == income_seg]
    if seg_name != "All":
        filtered = filtered[filtered["segment_name"] == seg_name]
    if churn_filter == "At Risk":
        filtered = filtered[filtered["predicted_churned"] == 1]
    elif churn_filter == "Safe":
        filtered = filtered[filtered["predicted_churned"] == 0]
    filtered = filtered[(filtered["ltv_score"] >= min_ltv) & (filtered["ltv_score"] <= max_ltv)]
    return filtered

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/dashboard")
def dashboard():
    df = apply_filters(df_raw, request.args)

    total = len(df)
    churned_count = int(df["predicted_churned"].sum())
    avg_ltv = round(float(df["ltv_score"].mean()), 1)
    avg_spent = round(float(df["total_spent"].mean()), 0)
    avg_churn_prob = round(float(df["predicted_churn_prob"].mean()) * 100, 1)

    # Segments breakdown
    seg_counts = df["segment_name"].value_counts().to_dict()
    seg_avg_ltv = df.groupby("segment_name")["ltv_score"].mean().round(1).to_dict()
    seg_avg_spent = df.groupby("segment_name")["total_spent"].mean().round(0).to_dict()

    # Revenue by region
    rev_region = df.groupby("region")["total_spent"].sum().round(0).to_dict()

    # Churn by income segment
    churn_income = df.groupby("income_segment")["predicted_churned"].mean().round(3).mul(100).round(1).to_dict()

    # LTV distribution buckets
    ltv_bins = pd.cut(df["ltv_score"], bins=[0,25,50,75,100], labels=["0-25","25-50","50-75","75-100"])
    ltv_dist = ltv_bins.value_counts().sort_index().to_dict()
    ltv_dist = {str(k): int(v) for k, v in ltv_dist.items()}

    # Monthly spend trend (simulated from data)
    monthly_trend = df.groupby(pd.cut(df["last_purchase_days"],
        bins=[0,30,60,120,180,270,365],
        labels=["<30d","30-60d","60-120d","120-180d","180-270d","270-365d"])
    )["total_spent"].mean().round(0).to_dict()
    monthly_trend = {str(k): float(v) for k, v in monthly_trend.items() if not pd.isna(v)}

    return jsonify({
        "summary": {
            "total_customers": total,
            "churned_count": churned_count,
            "churn_rate": round(churned_count / total * 100, 1) if total else 0,
            "avg_ltv": avg_ltv,
            "avg_spent": avg_spent,
            "avg_churn_prob": avg_churn_prob,
            "model_accuracy": churn_accuracy
        },
        "segments": {"counts": seg_counts, "avg_ltv": seg_avg_ltv, "avg_spent": seg_avg_spent},
        "revenue_by_region": rev_region,
        "churn_by_income": churn_income,
        "ltv_distribution": ltv_dist,
        "recency_trend": monthly_trend
    })

@app.route("/api/customers")
def customers():
    df = apply_filters(df_raw, request.args)
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 20))
    start = (page - 1) * per_page
    end = start + per_page

    cols = ["customer_id","name","age","region","income","income_segment",
            "total_purchases","avg_order_value","total_spent","last_purchase_days",
            "preferred_category","ltv_score","segment_name","predicted_churn_prob","predicted_churned","churned"]

    subset = df[cols].iloc[start:end]
    records = subset.to_dict(orient="records")

    return jsonify({
        "customers": records,
        "total": len(df),
        "page": page,
        "pages": (len(df) + per_page - 1) // per_page
    })

@app.route("/api/customer/<customer_id>")
def customer_detail(customer_id):
    row = df_raw[df_raw["customer_id"] == customer_id]
    if row.empty:
        return jsonify({"error": "Not found"}), 404

    r = row.iloc[0]
    recs = get_recommendations(r)

    return jsonify({
        "customer": {
            "customer_id": r["customer_id"],
            "name": r["name"],
            "age": int(r["age"]),
            "region": r["region"],
            "income": float(r["income"]),
            "income_segment": r["income_segment"],
            "total_purchases": int(r["total_purchases"]),
            "avg_order_value": float(r["avg_order_value"]),
            "total_spent": float(r["total_spent"]),
            "last_purchase_days": int(r["last_purchase_days"]),
            "preferred_category": r["preferred_category"],
            "recent_purchases": json.loads(r["recent_purchases"]),
            "loyalty_years": float(r["loyalty_years"]),
            "support_tickets": int(r["support_tickets"]),
            "email_open_rate": float(r["email_open_rate"]),
            "app_sessions_monthly": int(r["app_sessions_monthly"]),
            "ltv_score": float(r["ltv_score"]),
            "recency_score": float(r["recency_score"]),
            "frequency_score": float(r["frequency_score"]),
            "monetary_score": float(r["monetary_score"]),
            "segment_name": r["segment_name"],
            "predicted_churn_prob": float(r["predicted_churn_prob"]),
            "predicted_churned": int(r["predicted_churned"])
        },
        "recommendations": recs
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    le = encoders["income_segment"]
    income_enc = le.transform([data.get("income_segment","Medium")])[0]

    features = np.array([[
        float(data.get("age", 35)),
        float(data.get("recency_score", 50)),
        float(data.get("frequency_score", 30)),
        float(data.get("monetary_score", 40)),
        float(data.get("support_tickets", 2)),
        float(data.get("email_open_rate", 0.4)),
        float(data.get("app_sessions_monthly", 10)),
        float(data.get("loyalty_years", 1)),
        float(data.get("avg_order_value", 1500)),
        float(income_enc)
    ]])

    prob = float(models["churn"].predict_proba(features)[0][1])
    prediction = int(models["churn"].predict(features)[0])

    ltv = round(
        0.4 * float(data.get("monetary_score", 40)) +
        0.3 * float(data.get("frequency_score", 30)) +
        0.2 * float(data.get("recency_score", 50)) +
        0.1 * (float(data.get("loyalty_years", 1)) / 8 * 100), 1)

    risk = "High" if prob > 0.6 else "Medium" if prob > 0.35 else "Low"

    class FakeRow:
        def __init__(self, d): self._d = d
        def __getitem__(self, k): return self._d[k]
    fake_row = FakeRow({
        "recent_purchases": data.get("recent_purchases", '["Electronics"]'),
        "preferred_category": data.get("preferred_category", "Electronics"),
        "income_segment": data.get("income_segment", "Medium"),
        "ltv_score": ltv
    })
    recs = get_recommendations(fake_row)

    return jsonify({
        "churn_probability": round(prob * 100, 1),
        "churn_prediction": prediction,
        "risk_level": risk,
        "ltv_score": ltv,
        "model_accuracy": churn_accuracy,
        "recommendations": recs
    })

@app.route("/api/model_stats")
def model_stats():
    report = churn_report
    return jsonify({
        "accuracy": churn_accuracy,
        "precision_0": round(report["0"]["precision"] * 100, 1),
        "recall_0": round(report["0"]["recall"] * 100, 1),
        "precision_1": round(report["1"]["precision"] * 100, 1),
        "recall_1": round(report["1"]["recall"] * 100, 1),
        "f1_0": round(report["0"]["f1-score"] * 100, 1),
        "f1_1": round(report["1"]["f1-score"] * 100, 1),
        "support_0": int(report["0"]["support"]),
        "support_1": int(report["1"]["support"])
    })

@app.route("/api/filters_meta")
def filters_meta():
    return jsonify({
        "regions": ["All"] + sorted(df_raw["region"].unique().tolist()),
        "income_segments": ["All", "Low", "Medium", "High"],
        "segments": ["All"] + sorted(df_raw["segment_name"].unique().tolist()),
        "churn_options": ["All", "At Risk", "Safe"]
    })

if __name__ == "__main__":
    load_and_train()
    print("\n🚀 Server running at http://127.0.0.1:5000\n")
    app.run(debug=False, port=5000)
