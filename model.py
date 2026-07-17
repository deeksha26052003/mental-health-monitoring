import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from joblib import dump

# Load data
data = pd.read_csv('mental_health_data.csv')

# Encode categoricals
le_dict = {}
cat_cols = ['Gender', 'Sleep Schedule',
            'Exercise Frequency', 'Substance Use', 'Mood']

for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    le_dict[col] = le

# Feature engineering — combined mental health score
data['mental_health_score'] = (
    data['Do you have Anxiety?'] +
    data['Do you have Panic attack?'] +
    data['Stress Levels'] / 10
)

# Features + target (binary: depression yes/no)
features = ['Age', 'Gender', 'Daily Screen Time (hours)',
            'Do you have Anxiety?', 'Do you have Panic attack?',
            'Sleep Schedule', 'Exercise Frequency',
            'Family History of Mental Illness',
            'Substance Use', 'Stress Levels', 'Work-Life Balance',
            'Mood', 'mental_health_score']

X = data[features]
y = data['Do you have Depression?']  # binary 0/1

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE to fix class imbalance
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

model = XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.01,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=3,
    gamma=0.1,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train_res, y_train_res)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {acc * 100:.2f}%')
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['No Depression', 'Depression']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
imp = pd.Series(model.feature_importances_, index=features)
print("\nTop Features:")
print(imp.sort_values(ascending=False).head(5))

# Save
dump(model, 'mental_health_model.joblib')
dump(le_dict, 'label_encoders.joblib')
print("\nModel saved!")
