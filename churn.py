# importing the libraries required
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier


# 1. Load and clean

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# TotalCharges has blank strings for brand-new customers (tenure == 0)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
df["Churn"] = (df["Churn"] == "Yes").astype(int)
df = df.drop(columns=["customerID"])

print("Shape:", df.shape)
print("Churn rate: %.1f%%" % (100 * df["Churn"].mean()))

# 2. Encode features

y = df["Churn"]
X = pd.get_dummies(df.drop(columns=["Churn"]), drop_first=True).astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Models (class imbalance)
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, class_weight="balanced",
        random_state=42, n_jobs=-1,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=pos_weight,
        eval_metric="logloss", random_state=42,
    ),
}

probs = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]
    probs[name] = p
    pred = (p >= 0.5).astype(int)
    print("\n=== %s ===" % name)
    print("ROC-AUC: %.3f" % roc_auc_score(y_test, p))
    print(f"Confusion matrix:\n{confusion_matrix(y_test, pred)}")
    print(f"Classiffication report:\n{classification_report(y_test, pred, target_names=["Stay", "Churn"])}")

# 4. Feature importance ranking

lr = models["Logistic Regression"][-1]
importances = pd.DataFrame({
    "Logistic Regression (|coef|)": pd.Series(np.abs(lr.coef_[0]), index=X.columns),
    "Random Forest": pd.Series(models["Random Forest"].feature_importances_, index=X.columns),
    "XGBoost": pd.Series(models["XGBoost"].feature_importances_, index=X.columns),
})
# normalise each column to sum to 1 so models are comparable
importances = importances / importances.sum()
importances["Average"] = importances.mean(axis=1)
importances = importances.sort_values("Average", ascending=False)
print("\nTop 15 features (normalised importance):")
print(importances.head(15).round(3))

top = importances.head(15)["Average"][::-1]
plt.figure(figsize=(8, 6))
top.plot(kind="barh", color="royalblue")
plt.xlabel("Average normalised importance")
plt.title("Top churn drivers")
plt.tight_layout()
plt.show()

# 5. Risk scoring: who should the retention team call first?

best = models["XGBoost"]
scored = df.loc[X_test.index].copy()
scored["churn_prob"] = probs["XGBoost"]
scored["risk"] = pd.cut(scored["churn_prob"], [0, 0.4, 0.7, 1.0],
                        labels=["Low", "Medium", "High"], include_lowest=True)
print("\nRisk tiers (test set):")
print(scored.groupby("risk", observed=True)["Churn"].agg(["count", "mean"]).round(3))

high = scored[scored["risk"] == "High"]
print("\nHigh-risk segment profile vs. everyone else:")
for col in ["Contract", "PaymentMethod", "InternetService", "TechSupport"]:
    print("\n", col)
    print(pd.concat([
        high[col].value_counts(normalize=True).rename("High-risk"),
        scored[col].value_counts(normalize=True).rename("All"),
    ], axis=1).round(2))
print("\nMean tenure  - high-risk: %.1f | all: %.1f" % (high["tenure"].mean(), scored["tenure"].mean()))
print("Mean monthly - high-risk: %.1f | all: %.1f" % (high["MonthlyCharges"].mean(), scored["MonthlyCharges"].mean()))
