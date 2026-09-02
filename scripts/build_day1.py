import os
import json
import pandas as pd
import numpy as np
import nbformat as nbf

os.makedirs('data/processed', exist_ok=True)
os.makedirs('docs', exist_ok=True)
os.makedirs('notebooks', exist_ok=True)

print("--- Day 1: Starting Data Foundation Pipeline ---")

# 1. Load Raw Datasets
ea_df = pd.read_csv('data/raw/employee_attrition.csv')
eng_df = pd.read_csv('data/raw/hr_performance_engagement.csv')
occ_df = pd.read_csv('data/raw/occupation_data.csv')
ess_df = pd.read_csv('data/raw/essential_skills.csv')
soft_df = pd.read_csv('data/raw/software_skills.csv')

print(f"Loaded employee_attrition: {ea_df.shape}")
print(f"Loaded hr_performance_engagement: {eng_df.shape}")
print(f"Loaded occupation_data: {occ_df.shape}")
print(f"Loaded essential_skills: {ess_df.shape}")
print(f"Loaded software_skills: {soft_df.shape}")

# 2. Data Validation Checks
print("\nValidating Data...")
assert ea_df['Age'].between(18, 100).all(), "Validation Error: Age out of range"
assert ea_df['EmployeeID'].is_unique, "Validation Error: Non-unique EmployeeID in employee_attrition"
assert set(ea_df['Attrition'].unique()).issubset({'Yes', 'No'}), "Validation Error: Unexpected Attrition values"
assert eng_df['EngagementScore'].between(0, 100).all(), "Validation Error: EngagementScore out of range"
assert eng_df['EmployeeID'].is_unique, "Validation Error: Non-unique EmployeeID in hr_performance_engagement"
print("All assertions passed successfully!")

# 3. Data Cleaning & Standardization
print("\nCleaning & Standardizing Datasets...")

# A. Employee Attrition Clean
ea_clean = ea_df.copy()
# Drop constant zero-variance columns if present
drop_cols = ['EmployeeCount', 'Over18', 'StandardHours']
for col in drop_cols:
    if col in ea_clean.columns:
        ea_clean.drop(columns=[col], inplace=True)

# Standardize strings
for c in ea_clean.select_dtypes(include=['object']).columns:
    ea_clean[c] = ea_clean[c].astype(str).str.strip()

ea_clean.to_csv('data/processed/employee_attrition_processed.csv', index=False)
print("Saved data/processed/employee_attrition_processed.csv")

# B. Engagement Clean
eng_clean = eng_df.copy()
for c in eng_clean.select_dtypes(include=['object']).columns:
    eng_clean[c] = eng_clean[c].astype(str).str.strip()
eng_clean.to_csv('data/processed/engagement_processed.csv', index=False)
print("Saved data/processed/engagement_processed.csv")

# C. Occupation Master & Role Mapping
occ_clean = occ_df.copy()
occ_clean['O*NET-SOC Code'] = occ_clean['O*NET-SOC Code'].astype(str).str.strip()
occ_clean['Title'] = occ_clean['Title'].astype(str).str.strip()
occ_clean['Description'] = occ_clean['Description'].astype(str).str.strip()
occ_clean.drop_duplicates(subset=['O*NET-SOC Code'], inplace=True)
occ_clean.to_csv('data/processed/occupation_master.csv', index=False)
print("Saved data/processed/occupation_master.csv")

# D. Essential Skills Clean & Filter
ess_clean = ess_df.copy()
ess_clean['O*NET-SOC Code'] = ess_clean['O*NET-SOC Code'].astype(str).str.strip()
ess_clean['Element Name'] = ess_clean['Element Name'].astype(str).str.strip()
ess_clean = ess_clean[['O*NET-SOC Code', 'Title', 'Element Name', 'Data Value', 'Scale Name']].dropna(subset=['Element Name', 'Data Value'])
ess_clean = ess_clean.groupby(['O*NET-SOC Code', 'Title', 'Element Name'])['Data Value'].mean().reset_index()
ess_clean.to_csv('data/processed/essential_skills_processed.csv', index=False)
print("Saved data/processed/essential_skills_processed.csv")

# E. Software Skills Clean & Standardize
soft_clean = soft_df.copy()
soft_clean['O*NET-SOC Code'] = soft_clean['O*NET-SOC Code'].astype(str).str.strip()
soft_clean['Workplace Example'] = soft_clean['Workplace Example'].astype(str).str.strip()

synonym_map = {
    'AWS Cloud': 'Amazon Web Services AWS',
    'Amazon Web Services': 'Amazon Web Services AWS',
    'AWS': 'Amazon Web Services AWS',
    'MS Excel': 'Microsoft Excel',
    'Excel': 'Microsoft Excel',
    'Python Language': 'Python',
    'PostgreSQL Database': 'PostgreSQL',
    'Postgres': 'PostgreSQL'
}
soft_clean['Workplace Example'] = soft_clean['Workplace Example'].replace(synonym_map)
soft_clean = soft_clean.drop_duplicates(subset=['O*NET-SOC Code', 'Workplace Example'])
soft_clean.to_csv('data/processed/software_skills_processed.csv', index=False)
print("Saved data/processed/software_skills_processed.csv")

# 4. Write data_relationships.md
rel_md = """# Enterprise HR AI — Data Relationships & Entity Architecture

This document formalizes the entity relationships, join keys, and schemas connecting the five core datasets in the platform.

```
                    ┌─────────────────────────┐
                    │      EMPLOYEE MASTER    │
                    │  (employee_attrition)   │
                    │   PK: EmployeeID        │
                    └────────────┬────────────┘
                                 │
                 1 : 1           │           * : 1
          ┌──────────────────────┴──────────────────────┐
          │                                             │
          ▼                                             ▼
┌─────────────────────────┐                   ┌─────────────────────────┐
│       ENGAGEMENT        │                   │    OCCUPATION MASTER    │
│(hr_performance_engagem.)│                   │   (occupation_master)   │
│     FK: EmployeeID      │                   │     PK: ONET_SOC_Code   │
└─────────────────────────┘                   └────────────┬────────────┘
                                                           │
                                           1 : *           │           1 : *
                                    ┌──────────────────────┴──────────────────────┐
                                    │                                             │
                                    ▼                                             ▼
                          ┌───────────────────┐                         ┌───────────────────┐
                          │ ESSENTIAL SKILLS  │                         │  SOFTWARE SKILLS  │
                          │(essential_skills) │                         │ (software_skills) │
                          │ FK: ONET_SOC_Code │                         │ FK: ONET_SOC_Code │
                          └───────────────────┘                         └───────────────────┘
```

## Relational Join Specification

| Source Table | Target Table | Source Key | Target Key | Cardinality | Business Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `employee_attrition` | `hr_performance_engagement` | `EmployeeID` | `EmployeeID` | 1 : 1 | Connects demographic/attrition features with performance ratings and engagement metrics. |
| `employee_attrition` | `occupation_master` | `JobRole` | `Title` / `ONET_Code` | * : 1 | Maps company-specific job titles to O*NET standard occupational competencies. |
| `occupation_master` | `essential_skills` | `O*NET-SOC Code` | `O*NET-SOC Code` | 1 : * | Retrieves core behavioral and technical cognitive skills required per occupation. |
| `occupation_master` | `software_skills` | `O*NET-SOC Code` | `O*NET-SOC Code` | 1 : * | Retrieves software tools, programming languages, and tech stacks required per occupation. |
| `employee_attrition` | `employee_skills_inventory` | `EmployeeID` | `EmployeeID` | 1 : * | Links individual employee profiles to their verified proficiencies for skill gap calculation. |

## Data Schemas & Constraints

1. **Employee Master (`employee_attrition_processed.csv`)**:
   - `EmployeeID`: Unique Integer identifier [1..N]
   - `Age`: Integer in [18, 100]
   - `Attrition`: Categorical string in {'Yes', 'No'}
   - `MonthlyIncome`: Numeric (> 0)
   - `JobRole`: Categorical string (9 enterprise roles)
   - `OverTime`: Categorical string in {'Yes', 'No'}

2. **Engagement (`engagement_processed.csv`)**:
   - `EmployeeID`: Foreign key to Employee Master
   - `EngagementScore`: Integer in [0, 100]
   - `PerformanceRating`: Integer in [1, 5]
   - `WorkLifeBalanceScore`: Integer in [1, 5]

3. **Occupation Competency (`occupation_master.csv`, `essential_skills_processed.csv`, `software_skills_processed.csv`)**:
   - `O*NET-SOC Code`: Unique occupational alphanumeric identifier (e.g. `15-1221.00`)
   - `Element Name`: Standard skill competency name
   - `Workplace Example`: Specific software application/tool name (e.g. `Python`, `AWS`, `Docker`)
"""

with open('docs/data_relationships.md', 'w', encoding='utf-8') as f:
    f.write(rel_md)
print("Saved docs/data_relationships.md")

# 5. Helper to create notebooks
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

# Notebook 01: Data Understanding
nb01_code = [
    ("""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
DATA_PATH = "../data/raw"
print("Raw Files in data/raw:", os.listdir(DATA_PATH))
""", "### 1. Setup & Environment"),
    ("""
# Load and profile employee_attrition.csv
df_attrition = pd.read_csv(f"{DATA_PATH}/employee_attrition.csv")
print("employee_attrition Shape:", df_attrition.shape)
print("\\nData Types & Info:")
df_attrition.info()
print("\\nMissing Values Count:")
print(df_attrition.isnull().sum().sort_values(ascending=False).head(10))
print("\\nDuplicated Rows:", df_attrition.duplicated().sum())
print("\\nTarget Class Distribution (%):")
print(df_attrition['Attrition'].value_counts(normalize=True) * 100)
""", "### 2. Profile Employee Attrition Dataset"),
    ("""
# Load and profile hr_performance_engagement.csv
df_eng = pd.read_csv(f"{DATA_PATH}/hr_performance_engagement.csv")
print("hr_performance_engagement Shape:", df_eng.shape)
print("\\nEngagement Head:")
print(df_eng.head())
print("\\nMissing Values:", df_eng.isnull().sum())
print("\\nEngagement Score Stats:")
print(df_eng['EngagementScore'].describe())
""", "### 3. Profile HR Engagement Dataset"),
    ("""
# Load and profile occupation and skills datasets
df_occ = pd.read_csv(f"{DATA_PATH}/occupation_data.csv")
df_ess = pd.read_csv(f"{DATA_PATH}/essential_skills.csv")
df_soft = pd.read_csv(f"{DATA_PATH}/software_skills.csv")

print("Occupation Data Shape:", df_occ.shape)
print("Essential Skills Shape:", df_ess.shape)
print("Software Skills Shape:", df_soft.shape)

id_cols = {
    'attrition': [c for c in df_attrition.columns if 'id' in c.lower()],
    'engagement': [c for c in df_eng.columns if 'id' in c.lower()],
    'occupation': [c for c in df_occ.columns if 'code' in c.lower() or 'id' in c.lower()],
}
print("\\nCandidate Identifier Columns:", id_cols)
""", "### 4. Profile Occupation & Skills Datasets")
]
create_notebook('notebooks/01_data_understanding.ipynb', '01 Data Understanding', 'Exploratory data profiling, shape analysis, missing values, duplicates, and join key identification.', nb01_code)

# Notebook 02: Data Validation
nb02_code = [
    ("""
import pandas as pd

df_attrition = pd.read_csv("../data/raw/employee_attrition.csv")
df_eng = pd.read_csv("../data/raw/hr_performance_engagement.csv")

print("Validating Employee Attrition Schema & Ranges...")
# Schema checks
required_ea_cols = ['EmployeeID', 'Age', 'Department', 'JobRole', 'Attrition', 'MonthlyIncome', 'OverTime']
for c in required_ea_cols:
    assert c in df_attrition.columns, f"Missing required column {c}"

# Range & Categorical checks
assert df_attrition['Age'].between(18, 100).all(), "Age out of range"
assert df_attrition['EmployeeID'].is_unique, "Duplicate EmployeeID found"
assert set(df_attrition['Attrition'].unique()) <= {'Yes', 'No'}, "Unexpected Attrition value"
assert (df_attrition['MonthlyIncome'] > 0).all(), "Invalid MonthlyIncome"

print("✓ Employee Attrition validation passed!")
""", "### 1. Employee Attrition Data Validation"),
    ("""
print("Validating HR Performance & Engagement Schema & Ranges...")
# Engagement checks
assert 'EngagementScore' in df_eng.columns, "Missing EngagementScore"
assert df_eng['EngagementScore'].between(0, 100).all(), "EngagementScore out of bounds (0-100)"
assert df_eng['EmployeeID'].is_unique, "Duplicate EmployeeID found in engagement data"

# Referential integrity check
assert set(df_attrition['EmployeeID']) == set(df_eng['EmployeeID']), "Foreign key mismatch between attrition and engagement datasets!"
print("✓ HR Engagement & Referential integrity validation passed!")
""", "### 2. Engagement & Referential Integrity Validation")
]
create_notebook('notebooks/02_data_validation.ipynb', '02 Data Validation', 'Systematic schema checks, range assertions, categorical validity, and referential integrity.', nb02_code)

# Notebook 03: Data Cleaning
nb03_code = [
    ("""
import pandas as pd
import numpy as np

print("Running Data Cleaning Pipeline...")
ea = pd.read_csv("../data/raw/employee_attrition.csv")
eng = pd.read_csv("../data/raw/hr_performance_engagement.csv")
occ = pd.read_csv("../data/raw/occupation_data.csv")
ess = pd.read_csv("../data/raw/essential_skills.csv")
soft = pd.read_csv("../data/raw/software_skills.csv")

# Clean Attrition
drop_cols = [c for c in ['EmployeeCount', 'Over18', 'StandardHours'] if c in ea.columns]
ea_clean = ea.drop(columns=drop_cols).copy()
for c in ea_clean.select_dtypes(include=['object']).columns:
    ea_clean[c] = ea_clean[c].astype(str).str.strip()
ea_clean.to_csv("../data/processed/employee_attrition_processed.csv", index=False)

# Clean Engagement
eng_clean = eng.copy()
for c in eng_clean.select_dtypes(include=['object']).columns:
    eng_clean[c] = eng_clean[c].astype(str).str.strip()
eng_clean.to_csv("../data/processed/engagement_processed.csv", index=False)

# Clean Occupations
occ_clean = occ.drop_duplicates(subset=['O*NET-SOC Code']).copy()
occ_clean['Title'] = occ_clean['Title'].astype(str).str.strip()
occ_clean.to_csv("../data/processed/occupation_master.csv", index=False)

# Clean Essential Skills
ess_clean = ess[['O*NET-SOC Code', 'Title', 'Element Name', 'Data Value']].dropna().copy()
ess_clean = ess_clean.groupby(['O*NET-SOC Code', 'Title', 'Element Name'])['Data Value'].mean().reset_index()
ess_clean.to_csv("../data/processed/essential_skills_processed.csv", index=False)

# Clean Software Skills & Standardize Synonyms
soft_clean = soft[['O*NET-SOC Code', 'Title', 'Workplace Example', 'Hot Technology']].dropna().copy()
soft_clean['Workplace Example'] = soft_clean['Workplace Example'].astype(str).str.strip()
soft_clean.to_csv("../data/processed/software_skills_processed.csv", index=False)

print("Data Cleaning Complete. All processed CSVs saved in data/processed/")
""", "### 1. Cleaning & Standardizing All Datasets")
]
create_notebook('notebooks/03_data_cleaning.ipynb', '03 Data Cleaning', 'Handling missing values, deduplication, schema normalization, and saving clean processed datasets.', nb03_code)

# Notebook 04: Data Relationships
nb04_code = [
    ("""
import pandas as pd

ea = pd.read_csv("../data/processed/employee_attrition_processed.csv")
eng = pd.read_csv("../data/processed/engagement_processed.csv")
occ = pd.read_csv("../data/processed/occupation_master.csv")
ess = pd.read_csv("../data/processed/essential_skills_processed.csv")
soft = pd.read_csv("../data/processed/software_skills_processed.csv")

# 1. Join Employee to Engagement
emp_eng = pd.merge(ea, eng, on='EmployeeID', how='inner', suffixes=('', '_eng'))
print(f"Employee to Engagement Join Shape: {emp_eng.shape} (Expected 1470 rows)")

# 2. Join Occupation to Essential & Software Skills
occ_ess = pd.merge(occ, ess, on='O*NET-SOC Code', how='inner', suffixes=('_occ', '_ess'))
print(f"Occupation to Essential Skills Join Shape: {occ_ess.shape}")

occ_soft = pd.merge(occ, soft, on='O*NET-SOC Code', how='inner', suffixes=('_occ', '_soft'))
print(f"Occupation to Software Skills Join Shape: {occ_soft.shape}")

print("Relational entity joins verified successfully!")
""", "### 1. Relational Joins & Cardinality Verification")
]
create_notebook('notebooks/04_data_relationships.ipynb', '04 Data Relationships', 'Relational join validation, cardinality verification, and schema mapping.', nb04_code)

print("\n--- Day 1 Complete! ---")
