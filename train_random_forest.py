import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data

df = pd.read_csv("fight_features.csv")
print("Columns in dataset:", df.columns.tolist())

#  Handle Boolean Columns

bool_cols = ["started_adjacent", "move_forced"]
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False})

# Identify Target Column

TARGET_COL = "actor"  # winner column in fight_features.csv

# Drop Non-Feature Columns

drop_cols = [TARGET_COL]
if "fight_id" in df.columns:
    drop_cols.append("fight_id")
X = df.drop(columns=drop_cols)
y = df[TARGET_COL]

# Remove action_1 to action_5 before training

action_cols = [f"action_{i}" for i in range(1,6) if f"action_{i}" in X.columns]
X_model = X.drop(columns=action_cols)
print(f"Dropping columns from model: {action_cols}")

# Encode Categorical Features (excluding boolean columns)

cat_cols = X_model.select_dtypes(include=['object']).columns
label_encoders = {}

for col in cat_cols:
    X_model[col] = X_model[col].astype(str)
    le = LabelEncoder()
    X_model[col] = le.fit_transform(X_model[col])
    label_encoders[col] = le

# Train/Test Split (75% Train / 25% Test)

X_train, X_test, y_train, y_test = train_test_split(
    X_model, y, test_size=0.25, random_state=42
)

# Train Random Forest Classifier

rf_model = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_split=2,
    random_state=42
)
rf_model.fit(X_train, y_train)

#  Evaluate Model

y_pred = rf_model.predict(X_test)

print("\n=== Accuracy ===")
print(accuracy_score(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

#  Cross-validation

cv_scores = cross_val_score(rf_model, X_model, y, cv=5)
print("\n=== 5-Fold Cross-Validation Scores ===")
print(cv_scores)
print("Mean CV Score:", cv_scores.mean())
print("Std Dev:", cv_scores.std())

#  Permutation Importance

perm_importance = permutation_importance(
    rf_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)

importance_df = pd.DataFrame({
    "feature": X_test.columns,
    "importance": perm_importance.importances_mean
}).sort_values("importance", ascending=False)

print("\n=== Permutation Importance ===")
print(importance_df)

plt.figure(figsize=(14,6))
plt.bar(importance_df["feature"], importance_df["importance"])
plt.xticks(rotation=45, ha="right")
plt.title("Permutation Feature Importance (excluding actions 1-5)")
plt.ylabel("Importance (Mean Accuracy Drop)")
plt.tight_layout()
plt.show()
