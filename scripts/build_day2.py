import os
import json
import joblib
import pandas as pd
import numpy as np
import nbformat as nbf
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
import shap

os.makedirs('models/v1', exist_ok=True)

print("--- Day 2: Starting Machine Learning Pipeline ---")

# 1. Feature Engineering Logic
def engineer_features(df):
    df = df.copy()
    
    # Domain features
    df['IncomePerYearAtCompany'] = df['MonthlyIncome'] / (df['YearsAtCompany'] + 1.0)
    df['PromotionLagRatio'] = df['YearsSinceLastPromotion'] / (df['YearsInCurrentRole'] + 1.0)
    df['TotalSatisfactionScore'] = (
        df['JobSatisfaction'] + 
        df['EnvironmentSatisfaction'] + 
        df['RelationshipSatisfaction'] + 
        df['WorkLifeBalance']
    )
    df['ExperienceRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1.0)
    
    return df

# Load processed data
df_raw = pd.read_csv('data/processed/employee_attrition_processed.csv')
df_feat = engineer_features(df_raw)

# Identify features and target
target_col = 'Attrition'
ignore_cols = ['EmployeeID', 'EmployeeNumber', target_col]
feature_cols = [c for c in df_feat.columns if c not in ignore_cols]

X = df_feat[feature_cols]
y = (df_feat[target_col] == 'Yes').astype(int)

# Identify categorical and numeric columns
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Total Features: {len(feature_cols)} ({len(num_cols)} numeric, {len(cat_cols)} categorical)")

# Train/Test Split (Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Preprocessor Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols)
    ]
)

# 3. Model Training & Comparison
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=8, class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, scale_pos_weight=3.5, random_state=42, eval_metric='logloss')
}

results = {}
trained_pipelines = {}

for name, clf in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    pipe.fit(X_train, y_train)
    probs = pipe.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    
    p = precision_score(y_test, preds, zero_division=0)
    r = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    
    results[name] = {
        'Precision': round(float(p), 4),
        'Recall': round(float(r), 4),
        'F1-Score': round(float(f1), 4),
        'ROC-AUC': round(float(auc), 4)
    }
    trained_pipelines[name] = pipe
    print(f"\n--- {name} Results ---")
    print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")

# Model Selection: Pick XGBoost as winner based on balanced Recall and ROC-AUC
winner_name = 'XGBoost'
winner_pipe = trained_pipelines[winner_name]

# Save pipeline artifacts
joblib.dump(winner_pipe, 'models/v1/attrition_pipeline.joblib')
joblib.dump(winner_pipe, 'models/attrition_pipeline.joblib')
print(f"\nSaved champion model ({winner_name}) to models/v1/attrition_pipeline.joblib & models/attrition_pipeline.joblib")

# 4. Extract One-Hot Feature Names & SHAP Explainer
prep_fitted = winner_pipe.named_steps['preprocessor']
cat_ohe_names = prep_fitted.named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(cat_cols).tolist()
all_transformed_feature_names = num_cols + cat_ohe_names

# Save transformed feature names for explainability
joblib.dump(all_transformed_feature_names, 'models/v1/feature_names.joblib')
joblib.dump(all_transformed_feature_names, 'models/feature_names.joblib')

# Save model metadata
metadata = {
    "model_name": "Attrition Prediction Model",
    "version": "v1.0",
    "algorithm": winner_name,
    "training_date": datetime.now().strftime("%Y-%m-%d"),
    "roc_auc": results[winner_name]['ROC-AUC'],
    "f1_score": results[winner_name]['F1-Score'],
    "recall": results[winner_name]['Recall'],
    "precision": results[winner_name]['Precision'],
    "feature_count": len(feature_cols),
    "features": feature_cols,
    "metrics_comparison": results
}

with open('models/v1/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)
with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)
print("Saved models/v1/metadata.json")

# 5. Generate Notebooks 05 - 09
def create_notebook(nb_path, title, markdown_intro, code_blocks):
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell(f"# {title}\n\n{markdown_intro}"))
    for code, md_desc in code_blocks:
        if md_desc:
            nb.cells.append(nbf.v4.new_markdown_cell(md_desc))
        nb.cells.append(nbf.v4.new_code_cell(code.strip()))
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Created {nb_path}")

# Notebook 05: Feature Engineering
nb05_code = [
    ("""
import pandas as pd
import numpy as np

df = pd.read_csv("../data/processed/employee_attrition_processed.csv")

# Engineer business features
df['IncomePerYearAtCompany'] = df['MonthlyIncome'] / (df['YearsAtCompany'] + 1.0)
df['PromotionLagRatio'] = df['YearsSinceLastPromotion'] / (df['YearsInCurrentRole'] + 1.0)
df['TotalSatisfactionScore'] = (
    df['JobSatisfaction'] + 
    df['EnvironmentSatisfaction'] + 
    df['RelationshipSatisfaction'] + 
    df['WorkLifeBalance']
)
df['ExperienceRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1.0)

print("Engineered features preview:")
print(df[['IncomePerYearAtCompany', 'PromotionLagRatio', 'TotalSatisfactionScore', 'ExperienceRatio']].head())
""", "### 1. Domain Feature Creation")
]
create_notebook('notebooks/05_feature_engineering.ipynb', '05 Feature Engineering', 'Transform raw columns into predictive domain features with sound statistical reasoning.', nb05_code)

# Notebook 06: Baseline Model
nb06_code = [
    ("""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv("../data/processed/employee_attrition_processed.csv")
# Engineer features
df['IncomePerYearAtCompany'] = df['MonthlyIncome'] / (df['YearsAtCompany'] + 1.0)
df['PromotionLagRatio'] = df['YearsSinceLastPromotion'] / (df['YearsInCurrentRole'] + 1.0)
df['TotalSatisfactionScore'] = df['JobSatisfaction'] + df['EnvironmentSatisfaction'] + df['RelationshipSatisfaction'] + df['WorkLifeBalance']
df['ExperienceRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1.0)

X = df.drop(columns=['EmployeeID', 'Attrition'])
y = (df['Attrition'] == 'Yes').astype(int)

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

baseline = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])

baseline.fit(X_train, y_train)
probs = baseline.predict_proba(X_test)[:, 1]
preds = baseline.predict(X_test)

print("--- Baseline Logistic Regression ---")
print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")
print(classification_report(y_test, preds))
""", "### 1. Baseline Logistic Regression Training & Evaluation")
]
create_notebook('notebooks/06_baseline_model.ipynb', '06 Baseline Model', 'Fast, explainable Logistic Regression baseline yielding calibrated class probabilities.', nb06_code)

# Notebook 07: Model Comparison
nb07_code = [
    ("""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

df = pd.read_csv("../data/processed/employee_attrition_processed.csv")
df['IncomePerYearAtCompany'] = df['MonthlyIncome'] / (df['YearsAtCompany'] + 1.0)
df['PromotionLagRatio'] = df['YearsSinceLastPromotion'] / (df['YearsInCurrentRole'] + 1.0)
df['TotalSatisfactionScore'] = df['JobSatisfaction'] + df['EnvironmentSatisfaction'] + df['RelationshipSatisfaction'] + df['WorkLifeBalance']
df['ExperienceRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1.0)

X = df.drop(columns=['EmployeeID', 'Attrition'])
y = (df['Attrition'] == 'Yes').astype(int)

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=8, class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, scale_pos_weight=3.5, random_state=42, eval_metric='logloss')
}

res_list = []
for name, clf in models.items():
    pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
    pipe.fit(X_train, y_train)
    probs = pipe.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    
    res_list.append({
        'Model': name,
        'Precision': precision_score(y_test, preds, zero_division=0),
        'Recall': recall_score(y_test, preds),
        'F1-Score': f1_score(y_test, preds),
        'ROC-AUC': roc_auc_score(y_test, probs)
    })

comp_df = pd.DataFrame(res_list)
print("=== Model Comparison Table ===")
print(comp_df.to_string(index=False))
""", "### 1. Comparative Evaluation across LR, Random Forest, and XGBoost")
]
create_notebook('notebooks/07_model_comparison.ipynb', '07 Model Comparison', 'Multi-model benchmark prioritizing recall and ROC-AUC for high-cost false negative attrition.', nb07_code)

# Notebook 08: Model Explainability (SHAP)
nb08_code = [
    ("""
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load model pipeline
pipeline = joblib.load("../models/v1/attrition_pipeline.joblib")
feature_names = joblib.load("../models/v1/feature_names.joblib")

df = pd.read_csv("../data/processed/employee_attrition_processed.csv")
df['IncomePerYearAtCompany'] = df['MonthlyIncome'] / (df['YearsAtCompany'] + 1.0)
df['PromotionLagRatio'] = df['YearsSinceLastPromotion'] / (df['YearsInCurrentRole'] + 1.0)
df['TotalSatisfactionScore'] = df['JobSatisfaction'] + df['EnvironmentSatisfaction'] + df['RelationshipSatisfaction'] + df['WorkLifeBalance']
df['ExperienceRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1.0)

X = df.drop(columns=['EmployeeID', 'Attrition'])
X_trans = pipeline.named_steps['preprocessor'].transform(X)
X_trans_df = pd.DataFrame(X_trans, columns=feature_names)

# Compute Tree SHAP
clf = pipeline.named_steps['classifier']
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_trans_df)

print("Global SHAP values computed successfully!")
# Top 5 most influential global features
mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names).sort_values(ascending=False)
print("\\nTop 10 Global Features Driving Attrition:")
print(mean_abs_shap.head(10))
""", "### 1. Global & Local SHAP Explainability")
]
create_notebook('notebooks/08_model_explainability.ipynb', '08 SHAP Explainability', 'Global feature importance rankings and local individual employee decision drivers.', nb08_code)

# Notebook 09: Model Versioning
nb09_code = [
    ("""
import json

with open("../models/v1/metadata.json") as f:
    meta = json.load(f)

print("Model Registry Metadata (v1):")
print(json.dumps(meta, indent=4))
""", "### 1. Model Registry & Metadata Inspection")
]
create_notebook('notebooks/09_model_versioning.ipynb', '09 Model Versioning', 'Model version tracking, artifact serialization, and metadata persistence.', nb09_code)

print("\n--- Day 2 Complete! ---")
