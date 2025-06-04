import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import joblib

df = pd.read_csv("data/feature_data.csv")
X = df.drop(columns=["label", "filename"])
y = df["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 8, 16, 32],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("Best Parameters:", grid.best_params_)

y_proba = best_model.predict_proba(X_test)[:, 1]
y_pred_custom = (y_proba >= 0.6).astype(int)

print("정확도:", accuracy_score(y_test, y_pred_custom))
print("분류 리포트:\n", classification_report(y_test, y_pred_custom))

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_custom), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

joblib.dump(best_model, "models/pneumonia_rf_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("모델 및 스케일러 저장 완료!")