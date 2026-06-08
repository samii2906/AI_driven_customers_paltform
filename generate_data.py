import pandas as pd
import numpy as np
import json

np.random.seed(42)
N = 350

names_first = ["Aarav","Ananya","Rohan","Priya","Vikram","Sneha","Arjun","Kavya","Rahul","Deepika",
    "Karthik","Meera","Aditya","Pooja","Suresh","Lakshmi","Ravi","Nisha","Sanjay","Divya",
    "Amit","Sonia","Rajesh","Neha","Vijay","Preeti","Arun","Swati","Nikhil","Asha",
    "Tarun","Shreya","Mohan","Rekha","Pavan","Anjali","Sunil","Madhuri","Girish","Usha",
    "Harish","Sunita","Prakash","Radha","Ganesh","Smita","Naveen","Geeta","Ramesh","Padma"]

names_last = ["Sharma","Reddy","Kumar","Verma","Patel","Nair","Singh","Rao","Gupta","Iyer",
    "Joshi","Pillai","Mehta","Sinha","Mishra","Chauhan","Agarwal","Pandey","Bhat","Desai"]

categories = ["Electronics","Clothing","Groceries","Beauty","Sports","Home & Kitchen","Books","Toys"]
regions = ["North","South","East","West","Central"]

customer_ids = [f"CUST{str(i).zfill(4)}" for i in range(1, N+1)]
full_names = [f"{np.random.choice(names_first)} {np.random.choice(names_last)}" for _ in range(N)]
ages = np.random.randint(22, 65, N)
incomes = np.round(np.random.choice(
    [np.random.uniform(15000,35000), np.random.uniform(35000,75000), np.random.uniform(75000,200000)],
    N), -2)
income_segment = pd.cut(incomes, bins=[0,35000,75000,200000], labels=["Low","Medium","High"])

total_purchases = np.random.randint(1, 60, N)
avg_order_value = np.round(np.random.uniform(200, 8000, N), 2)
total_spent = np.round(total_purchases * avg_order_value * np.random.uniform(0.8, 1.2, N), 2)
last_purchase_days = np.random.randint(1, 365, N)
recency_score = np.clip(100 - last_purchase_days * 0.27, 0, 100).round(1)
frequency_score = np.clip(total_purchases * 1.7, 0, 100).round(1)
monetary_score = np.clip(total_spent / 2000, 0, 100).round(1)

preferred_category = np.random.choice(categories, N)
region = np.random.choice(regions, N)
loyalty_years = np.round(np.random.uniform(0.1, 8, N), 1)
support_tickets = np.random.randint(0, 10, N)
email_open_rate = np.round(np.random.uniform(0, 1, N), 2)
app_sessions_monthly = np.random.randint(0, 50, N)

# Churn logic: high last_purchase_days, low frequency, many support tickets → churn
churn_prob = (
    0.3 * (last_purchase_days / 365) +
    0.25 * (1 - frequency_score / 100) +
    0.2 * (support_tickets / 10) +
    0.15 * (1 - email_open_rate) +
    0.1 * (1 - app_sessions_monthly / 50)
)
churn_prob = np.clip(churn_prob + np.random.normal(0, 0.05, N), 0, 1)
churned = (churn_prob > 0.55).astype(int)

# LTV score
ltv_score = np.round(
    0.4 * monetary_score +
    0.3 * frequency_score +
    0.2 * recency_score +
    0.1 * (loyalty_years / 8 * 100), 1
)

recent_purchases = []
for i in range(N):
    cats = np.random.choice(categories, size=np.random.randint(1,4), replace=False).tolist()
    recent_purchases.append(json.dumps(cats))

df = pd.DataFrame({
    "customer_id": customer_ids,
    "name": full_names,
    "age": ages,
    "region": region,
    "income": incomes,
    "income_segment": income_segment,
    "total_purchases": total_purchases,
    "avg_order_value": avg_order_value,
    "total_spent": total_spent,
    "last_purchase_days": last_purchase_days,
    "recency_score": recency_score,
    "frequency_score": frequency_score,
    "monetary_score": monetary_score,
    "preferred_category": preferred_category,
    "recent_purchases": recent_purchases,
    "loyalty_years": loyalty_years,
    "support_tickets": support_tickets,
    "email_open_rate": email_open_rate,
    "app_sessions_monthly": app_sessions_monthly,
    "ltv_score": ltv_score,
    "churn_probability": churn_prob.round(3),
    "churned": churned
})

df.to_csv("/home/claude/customer_analytics/customers.csv", index=False)
print(f"Dataset created: {len(df)} customers, {churned.sum()} churned ({churned.mean()*100:.1f}%)")
print(df.head(3).to_string())
