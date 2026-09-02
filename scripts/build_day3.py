import os
import json
import joblib
import pandas as pd
import numpy as np
import nbformat as nbf

print("--- Day 3: Starting Workforce Intelligence Pipeline ---")

# 1. Load Clean Processed Datasets
ea = pd.read_csv('data/processed/employee_attrition_processed.csv')
eng = pd.read_csv('data/processed/engagement_processed.csv')
occ = pd.read_csv('data/processed/occupation_master.csv')
ess = pd.read_csv('data/processed/essential_skills_processed.csv')
soft = pd.read_csv('data/processed/software_skills_processed.csv')

# Load trained attrition pipeline
pipeline = joblib.load('models/v1/attrition_pipeline.joblib')
feature_names = joblib.load('models/v1/feature_names.joblib')

# Role mapping to O*NET
role_to_onet = {
    'Sales Executive': '11-2022.00',
    'Research Scientist': '15-1221.00',
    'Laboratory Technician': '19-4031.00',
    'Manufacturing Director': '11-3051.00',
    'Healthcare Representative': '11-9111.00',
    'Manager': '11-1021.00',
    'Sales Representative': '41-4012.00',
    'Research Director': '11-9121.00',
    'Human Resources': '13-1071.00'
}

# 2. Build Role Requirements Dictionary
role_requirements = {}
for role, onet_code in role_to_onet.items():
    # Top 5 essential skills by value
    role_ess = ess[ess['O*NET-SOC Code'] == onet_code].sort_values(by='Data Value', ascending=False)['Element Name'].head(5).tolist()
    # Software tools
    role_soft = soft[soft['O*NET-SOC Code'] == onet_code]['Workplace Example'].head(6).tolist()
    
    # Combined required skill set
    combined = list(dict.fromkeys(role_ess + role_soft))
    role_requirements[role] = combined

print("Role Competency Requirements built for all 9 roles.")

# 3. Build Realistic Employee Current Skills Inventory
np.random.seed(42)
emp_skills_list = []

for idx, row in ea.iterrows():
    emp_id = int(row['EmployeeID'])
    role = row['JobRole']
    reqs = role_requirements.get(role, ['Problem Solving', 'Communication', 'Microsoft Excel'])
    
    # Employee possesses a subset of required skills + some general skills based on tenure and education
    # Senior / longer tenure employees possess more of the required skills
    tenure_prob = min(0.9, 0.45 + (row['YearsAtCompany'] / 25.0) + (row['JobLevel'] * 0.08))
    
    # Decide which skills they currently have
    num_skills_has = max(1, int(len(reqs) * np.random.uniform(tenure_prob - 0.2, min(1.0, tenure_prob + 0.15))))
    has_skills = list(np.random.choice(reqs, size=min(num_skills_has, len(reqs)), replace=False))
    
    for sk in has_skills:
        emp_skills_list.append({
            'EmployeeID': emp_id,
            'JobRole': role,
            'CurrentSkill': sk
        })

emp_skills_df = pd.DataFrame(emp_skills_list)
emp_skills_df.to_csv('data/processed/employee_skills_inventory.csv', index=False)
print(f"Saved data/processed/employee_skills_inventory.csv ({len(emp_skills_df)} skill entries)")

# 4. Skill Gap Engine (Set Difference)
emp_gaps = {}
for emp_id in ea['EmployeeID'].unique():
    role = ea.loc[ea['EmployeeID'] == emp_id, 'JobRole'].iloc[0]
    required = set(role_requirements.get(role, []))
    has = set(emp_skills_df[emp_skills_df['EmployeeID'] == emp_id]['CurrentSkill'].tolist())
    gap = sorted(list(required - has))
    emp_gaps[emp_id] = gap

# 5. Organization-Wide Skill Gap Rollup
all_gaps = []
for gaps in emp_gaps.values():
    all_gaps.extend(gaps)

org_gap_counts = pd.Series(all_gaps).value_counts().reset_index()
org_gap_counts.columns = ['Skill', 'MissingEmployeeCount']

def assign_severity(count):
    if count >= 100:
        return 'HIGH'
    elif count >= 50:
        return 'MEDIUM'
    else:
        return 'LOW'

org_gap_counts['Severity'] = org_gap_counts['MissingEmployeeCount'].apply(assign_severity)
org_gap_counts.to_csv('data/processed/organization_skill_gaps.csv', index=False)
print("Saved data/processed/organization_skill_gaps.csv")
print("\nTop Organization Skill Gaps:")
print(org_gap_counts.head(8))

# 6. Recommendation Engine (Curated Course Mapping)
course_catalog = {
    'Critical Thinking': 'Executive Decision Making & Critical Thinking Workshop',
    'Active Listening': 'Active Listening and High-Impact Communication for Leaders',
    'Complex Problem Solving': 'Structured Problem Solving and Root Cause Analysis',
    'Judgment and Decision Making': 'Strategic Judgment and Risk-Calibrated Decisions',
    'Speaking': 'Advanced Executive Presentation and Persuasive Speaking',
    'Writing': 'Business and Technical Writing Excellence',
    'Reading Comprehension': 'Information Synthesis and Rapid Document Analysis',
    'Monitoring': 'Performance Monitoring and KPI Management Masterclass',
    'Time Management': 'Agile Prioritization and Enterprise Time Mastery',
    'Management of Personnel Resources': 'Strategic Talent Leadership & People Operations',
    'Amazon Web Services AWS': 'AWS Certified Solutions Architect & Cloud Engineering Track',
    'Microsoft Excel': 'Advanced Financial Modeling & Data Analysis in Excel',
    'Python': 'Applied Python for Data Science and Machine Learning Automation',
    'PostgreSQL': 'Relational Database Architecture and SQL Mastery',
    'Docker': 'Containerization & Microservices with Docker and Kubernetes',
    'MLOps': 'End-to-End MLOps: Continuous Delivery for Machine Learning',
    'Salesforce': 'Salesforce CRM Enterprise Administration & Pipeline Optimization',
    'Tableau': 'Enterprise Visual Analytics & Dashboarding with Tableau',
    'R': 'Statistical Computing and Data Analysis with R',
    'SAP': 'SAP Enterprise Resource Planning Core Implementation'
}

def get_recommendation(skill_gaps):
    if not skill_gaps:
        return "Role Competencies Complete - Ready for Leadership Mentorship"
    top_gap = skill_gaps[0]
    course = course_catalog.get(top_gap, f"Advanced Certification in {top_gap}")
    return f"Recommended: {course} (Addresses critical gap: {top_gap})"

# 7. Compute Model Attrition Predictions & SHAP Drivers
# Engineer features for inference
ea_inf = ea.copy()
ea_inf['IncomePerYearAtCompany'] = ea_inf['MonthlyIncome'] / (ea_inf['YearsAtCompany'] + 1.0)
ea_inf['PromotionLagRatio'] = ea_inf['YearsSinceLastPromotion'] / (ea_inf['YearsInCurrentRole'] + 1.0)
ea_inf['TotalSatisfactionScore'] = ea_inf['JobSatisfaction'] + ea_inf['EnvironmentSatisfaction'] + ea_inf['RelationshipSatisfaction'] + ea_inf['WorkLifeBalance']
ea_inf['ExperienceRatio'] = ea_inf['YearsAtCompany'] / (ea_inf['TotalWorkingYears'] + 1.0)

X_all = ea_inf.drop(columns=['EmployeeID', 'Attrition'])
probs = pipeline.predict_proba(X_all)[:, 1]

# Extract feature names & contributions
clf = pipeline.named_steps['classifier']
prep = pipeline.named_steps['preprocessor']
X_trans = prep.transform(X_all)

# Get top drivers per individual
import shap
explainer = shap.TreeExplainer(clf)
shap_vals = explainer.shap_values(X_trans)

top_drivers_list = []
for i in range(len(ea)):
    row_shaps = shap_vals[i]
    top_indices = np.argsort(row_shaps)[-3:][::-1]
    drivers = [feature_names[idx].split('__')[-1] for idx in top_indices if row_shaps[idx] > 0]
    if not drivers:
        drivers = [feature_names[top_indices[0]].split('__')[-1]]
    top_drivers_list.append(", ".join(drivers))

# 8. Build Master Employee Intelligence Table
master_df = pd.DataFrame()
master_df['EmployeeID'] = ea['EmployeeID']
master_df['Department'] = ea['Department']
master_df['JobRole'] = ea['JobRole']
master_df['Age'] = ea['Age']
master_df['MonthlyIncome'] = ea['MonthlyIncome']
master_df['AttritionProbability'] = np.round(probs, 4)

def categorize_risk(p):
    if p >= 0.60:
        return 'HIGH'
    elif p >= 0.30:
        return 'MEDIUM'
    else:
        return 'LOW'

master_df['RiskLevel'] = master_df['AttritionProbability'].apply(categorize_risk)

# Merge engagement
master_df = master_df.merge(eng[['EmployeeID', 'EngagementScore', 'PerformanceRating', 'WorkLifeBalanceScore']], on='EmployeeID', how='left')

# Add Skill Gaps and Recommendations
master_df['SkillGaps'] = master_df['EmployeeID'].apply(lambda x: ", ".join(emp_gaps.get(x, [])))
master_df['GapCount'] = master_df['EmployeeID'].apply(lambda x: len(emp_gaps.get(x, [])))
master_df['UpskillingRecommendation'] = master_df['EmployeeID'].apply(lambda x: get_recommendation(emp_gaps.get(x, [])))
master_df['TopRiskDrivers'] = top_drivers_list

master_df.to_csv('data/processed/employee_intelligence_master.csv', index=False)
print("Saved data/processed/employee_intelligence_master.csv (1470 employees)")
print("\nMaster Intelligence Table Preview:")
print(master_df[['EmployeeID', 'Department', 'JobRole', 'AttritionProbability', 'RiskLevel', 'EngagementScore', 'SkillGaps', 'UpskillingRecommendation']].head())

# 9. Create Notebooks 10 - 16
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

# Notebook 10: Engagement Intelligence
nb10_code = [
    ("""
import pandas as pd
eng = pd.read_csv("../data/processed/engagement_processed.csv")
print("Average Engagement Score:", eng['EngagementScore'].mean())
print("\\nDepartment-wise Engagement Breakdown:")
dept_eng = eng.groupby('Department')['EngagementScore'].agg(['mean', 'median', 'std', 'count']).sort_values(by='mean', ascending=False)
print(dept_eng)

print("\\nLow Engagement Cohort (Score < 50):")
print(eng[eng['EngagementScore'] < 50][['EmployeeID', 'Department', 'JobRole', 'EngagementScore', 'PerformanceRating']].head(10))
""", "### 1. Engagement Analytics & Cohort Breakdown")
]
create_notebook('notebooks/10_engagement_intelligence.ipynb', '10 Engagement Intelligence', 'Engagement metrics aggregation by department, role, and low-engagement cohort analysis.', nb10_code)

# Notebook 11: Role Intelligence
nb11_code = [
    ("""
import pandas as pd
occ = pd.read_csv("../data/processed/occupation_master.csv")
ess = pd.read_csv("../data/processed/essential_skills_processed.csv")
soft = pd.read_csv("../data/processed/software_skills_processed.csv")

print(f"Master Occupations Count: {len(occ)}")
print("Sample Occupations with mapped skills:")
for code in ['15-1221.00', '11-2022.00', '19-4031.00']:
    title = occ[occ['O*NET-SOC Code'] == code]['Title'].iloc[0]
    ess_skills = ess[ess['O*NET-SOC Code'] == code]['Element Name'].head(4).tolist()
    tools = soft[soft['O*NET-SOC Code'] == code]['Workplace Example'].head(4).tolist()
    print(f"\\nRole: {title} ({code})")
    print(f"  Essential Skills: {ess_skills}")
    print(f"  Software/Tools: {tools}")
""", "### 1. Master Role Competencies & Tools Mapping")
]
create_notebook('notebooks/11_role_intelligence.ipynb', '11 Role Intelligence', 'Standardized master role catalog and competency profile mappings.', nb11_code)

# Notebook 12: Employee Skills
nb12_code = [
    ("""
import pandas as pd
emp_skills = pd.read_csv("../data/processed/employee_skills_inventory.csv")
print("Employee Skills Inventory Shape:", emp_skills.shape)
print("Skills per Employee Distribution:")
print(emp_skills.groupby('EmployeeID').size().describe())
print("\\nSample Employee Skills Record:")
print(emp_skills.head(10))
""", "### 1. Employee Skills Inventory Catalog")
]
create_notebook('notebooks/12_employee_skills.ipynb', '12 Employee Skills', 'Controlled inventory of current skills across the workforce.', nb12_code)

# Notebook 13: Skill Gap Engine
nb13_code = [
    ("""
import pandas as pd
master = pd.read_csv("../data/processed/employee_intelligence_master.csv")
print("Skill Gaps Sample:")
print(master[['EmployeeID', 'JobRole', 'SkillGaps', 'GapCount']].head(10))

print("\\nEmployees with 0 Skill Gaps:", (master['GapCount'] == 0).sum())
print("Average Gaps per Employee:", master['GapCount'].mean())
""", "### 1. Set-Subtraction Skill Gap Engine")
]
create_notebook('notebooks/13_skill_gap_engine.ipynb', '13 Skill Gap Engine', 'Set-difference mathematical engine calculating missing skills per employee.', nb13_code)

# Notebook 14: Organization Skill Gap
nb14_code = [
    ("""
import pandas as pd
org_gaps = pd.read_csv("../data/processed/organization_skill_gaps.csv")
print("=== Organization-Wide Critical Skill Gaps ===")
print(org_gaps)
""", "### 1. Enterprise Skill Gap Rollup & Severity Tiering")
]
create_notebook('notebooks/14_organization_skill_gap.ipynb', '14 Organization Skill Gap', 'Organization-wide skill gap aggregation with High/Medium/Low severity tiers.', nb14_code)

# Notebook 15: Recommendation Engine
nb15_code = [
    ("""
import pandas as pd
master = pd.read_csv("../data/processed/employee_intelligence_master.csv")
print("Targeted Upskilling Recommendations Preview:")
print(master[['EmployeeID', 'JobRole', 'SkillGaps', 'UpskillingRecommendation']].head(10))
""", "### 1. Upskilling Recommendations")
]
create_notebook('notebooks/15_recommendation_engine.ipynb', '15 Recommendation Engine', 'Personalized upskilling recommendations and learning pathways.', nb15_code)

# Notebook 16: Employee Intelligence Layer
nb16_code = [
    ("""
import pandas as pd
master = pd.read_csv("../data/processed/employee_intelligence_master.csv")
print("=== Unified Employee Intelligence Master Table ===")
print(f"Total Records: {len(master)}")
print(master.info())
print("\\nRisk Level Breakdown:")
print(master['RiskLevel'].value_counts())
""", "### 1. Unified Employee Intelligence Table")
]
create_notebook('notebooks/16_employee_intelligence.ipynb', '16 Employee Intelligence Layer', 'Consolidated enterprise workforce intelligence table uniting attrition, engagement, competencies, and learning.', nb16_code)

print("\n--- Day 3 Complete! ---")
