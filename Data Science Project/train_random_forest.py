import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load extracted features
df = pd.read_csv("fight_features.csv")

# Define features and target
feature_cols = [
    "A_hp", "A_ac", "A_attack_bonus", "A_speed", "A_dmg_main", "A_dmg_backup",
    "B_hp", "B_ac", "B_attack_bonus", "B_speed", "B_dmg_main", "B_dmg_backup",
    "starting_distance"
]

X = df[feature_cols]
y = df["winner"]

# Encode categorical columns (damage dice)
dice_cols = ["A_dmg_main", "A_dmg_backup", "B_dmg_main", "B_dmg_backup"]
label_encoders = {}

for col in dice_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Train/test split (75% train / 25% test, stratified by winner)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=400,
    random_state=42
)
rf_model.fit(X_train, y_train)

# Predict on test set
y_pred = rf_model.predict(X_test)

# Evaluation
print("=== Accuracy ===")
print(accuracy_score(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Feature Importance
importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\n=== Feature Importances ===")
print(importance_df)

plt.figure(figsize=(12,6))
plt.bar(importance_df["feature"], importance_df["importance"])
plt.xticks(rotation=45, ha="right")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()
