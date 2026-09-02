# Enterprise HR AI — Workforce Intelligence & Upskilling Platform

An enterprise-grade, end-to-end Machine Learning and Workforce Intelligence platform that predicts employee attrition risk, quantifies engagement, identifies organizational skill gaps, and recommends personalized learning pathways to upskill employees.

---

## 🏗️ System Architecture

```
                                  ┌───────────────────────────┐
                                  │      USER / HR LEADER     │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │   Streamlit Web UI (8501) │
                                  │   (Interactive Analytics) │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │    FastAPI Backend (8000) │
                                  │  (Pydantic Schema Guard)  │
                                  └──────┬──────────────┬─────┘
                                         │              │
                    ┌────────────────────┴───┐          └───┬────────────────────┐
                    ▼                        ▼              ▼                    ▼
       ┌─────────────────────────┐ ┌──────────────────┐ ┌───────────────┐ ┌───────────────┐
       │   Attrition ML Engine   │ │ Skill Gap Engine │ │ Engagement    │ │ Audit Logging │
       │  - XGBoost Classifier   │ │- Set Difference  │ │ Analytics     │ │ & Drift Watch │
       │  - SHAP Explainability  │ │- O*NET Standards │ │ (Dept Cohorts)│ │ (KS-test / PSI│
       └────────────┬────────────┘ └────────┬─────────┘ └───────┬───────┘ └───────┬───────┘
                    │                       │                   │                 │
                    └───────────────────────┴─────────┬─────────┴─────────────────┘
                                                      ▼
                                       ┌─────────────────────────────┐
                                       │ Unified Intelligence Master │
                                       │ (1470 Employees Dossiers)   │
                                       └─────────────────────────────┘
```

---

## 📁 Repository Structure

```
enterprise_hr_ai/
├── data/
│   ├── raw/                             # Raw untouched CSV sources
│   │   ├── employee_attrition.csv       # Demographic & operational employee records
│   │   ├── hr_performance_engagement.csv# Engagement scores & ratings
│   │   ├── occupation_data.csv          # Master O*NET occupation codes
│   │   ├── essential_skills.csv         # Core cognitive competencies
│   │   └── software_skills.csv          # Tools and software competencies
│   ├── processed/                       # Cleaned, standardized data artifacts
│   │   ├── employee_attrition_processed.csv
│   │   ├── engagement_processed.csv
│   │   ├── occupation_master.csv
│   │   ├── essential_skills_processed.csv
│   │   ├── software_skills_processed.csv
│   │   ├── employee_skills_inventory.csv
│   │   ├── organization_skill_gaps.csv
│   │   └── employee_intelligence_master.csv
│   └── predictions/                     # Production prediction audit logs
│       └── prediction_logs.csv
├── docs/
│   └── data_relationships.md           # Entity relationship model documentation
├── notebooks/                          # 16 Step-by-Step Executable Jupyter Notebooks
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_validation.ipynb
│   ├── 03_data_cleaning.ipynb
│   ├── 04_data_relationships.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_baseline_model.ipynb
│   ├── 07_model_comparison.ipynb
│   ├── 08_model_explainability.ipynb
│   ├── 09_model_versioning.ipynb
│   ├── 10_engagement_intelligence.ipynb
│   ├── 11_role_intelligence.ipynb
│   ├── 12_employee_skills.ipynb
│   ├── 13_skill_gap_engine.ipynb
│   ├── 14_organization_skill_gap.ipynb
│   ├── 15_recommendation_engine.ipynb
│   └── 16_employee_intelligence.ipynb
├── models/                             # Versioned Model Registry
│   ├── v1/
│   │   ├── attrition_pipeline.joblib
│   │   ├── feature_names.joblib
│   │   └── metadata.json
│   ├── attrition_pipeline.joblib
│   └── metadata.json
├── app/                                # FastAPI Production Backend
│   ├── main.py                         # Application factory & lifespan
│   ├── api/                            # REST route handlers
│   │   ├── attrition.py                # Single & batch prediction endpoints
│   │   ├── dashboard.py                # Aggregation & employee dossiers
│   │   └── skills.py                   # Skill gap endpoints
│   ├── services/                       # Business logic services
│   │   ├── attrition_service.py
│   │   ├── engagement_service.py
│   │   ├── skill_gap_service.py
│   │   └── recommendation_service.py
│   ├── validation/                     # Pydantic input schemas & validators
│   │   ├── employee_schema.py
│   │   └── engagement_schema.py
│   ├── ml/                             # Model loader & inference engine
│   │   ├── model_loader.py
│   │   └── predictor.py
│   └── utils/                          # Configurations & structured logging
│       ├── config.py
│       └── logger.py
├── frontend/                           # Streamlit Web Application
│   └── app.py
├── monitoring/                         # Drift Detection & Retraining Strategy
│   ├── drift_detector.py
│   └── retraining_policy.py
├── tests/                              # Pytest Unit & Integration Test Suite
│   ├── test_validation.py
│   ├── test_ml_pipeline.py
│   ├── test_skill_gap.py
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start & Setup

### 1. Environment Installation
```bash
# Clone or navigate to the project directory
cd Agentic_Project

# Create and activate virtual environment (optional)
python -m venv venv
source venv/bin/activate # or venv\Scripts\activate on Windows

# Install all required packages
pip install -r requirements.txt
```

### 2. Run Data Pipeline & Train Models
```bash
# Day 1: Data Foundation (Validation, Cleaning, Relationship docs)
python scripts/build_day1.py

# Day 2: Machine Learning (Feature Engineering, Comparison, SHAP, Versioning)
python scripts/build_day2.py

# Day 3: Workforce Intelligence (Skill Gaps, Rollup, Recommendation Engine)
python scripts/build_day3.py
```

### 3. Run Automated Tests
```bash
python -m pytest -v
```

---

## 🚀 Running the Services

### Running the FastAPI Backend
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation available at: `http://localhost:8000/docs`

### Running the Streamlit Dashboard
```bash
streamlit run frontend/app.py
```
Interactive Web Dashboard available at: `http://localhost:8501`

---

## 🐳 Docker Deployment

Run both the FastAPI backend and Streamlit dashboard simultaneously in Docker:
```bash
docker-compose up --build
```
- FastAPI: `http://localhost:8000`
- Streamlit UI: `http://localhost:8501`

---

## 🤖 Machine Learning & Explainability

### Model Benchmark Comparison
In workforce attrition, missing an employee who is about to quit (False Negative) is significantly more expensive than conducting an unnecessary retention check-in (False Positive). Hence, our model selection prioritizes **Recall and ROC-AUC**:

| Model | Precision | Recall | F1-Score | ROC-AUC | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** (Baseline) | 0.3704 | **0.6383** | **0.4688** | **0.8076** | Baseline |
| **Random Forest** | 0.4211 | 0.1702 | 0.2424 | 0.7795 | Evaluated |
| **XGBoost Classifier** | **0.5000** | 0.3830 | 0.4337 | 0.7632 | **Champion (Saved v1.0)** |

### Feature Engineering
1. `IncomePerYearAtCompany = MonthlyIncome / (YearsAtCompany + 1)`
2. `PromotionLagRatio = YearsSinceLastPromotion / (YearsInCurrentRole + 1)`
3. `TotalSatisfactionScore = JobSatisfaction + EnvironmentSatisfaction + RelationshipSatisfaction + WorkLifeBalance`
4. `ExperienceRatio = YearsAtCompany / (TotalWorkingYears + 1)`

### SHAP Decision Explainability
- **Global Feature Importance**: Identifies enterprise-wide drivers (OverTime, TotalSatisfactionScore, MonthlyIncome, PromotionLagRatio).
- **Local Individual Factors**: Provides the top 3 contributing factors per employee so HR business partners know *why* an employee is flagged.

---

## 🎯 Skill Gap Engine & Recommendation Logic

1. **Set Subtraction Logic**:
   $$\text{Skill Gap} = \text{Required Skills}_{\text{Job Role}} \setminus \text{Current Verified Skills}_{\text{Employee}}$$
2. **Organization-Wide Rollup**:
   - **HIGH Severity**: $\ge 100$ employees missing skill
   - **MEDIUM Severity**: $50 - 99$ employees missing skill
   - **LOW Severity**: $< 50$ employees missing skill
3. **Targeted Learning Pathways**: Direct mapping of identified skill deficits to curated enterprise workshops and certification courses.

---

## 📡 API Endpoints Specification

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health status check |
| `POST` | `/predict/attrition` | Predicts attrition probability, risk tier, and local SHAP drivers |
| `GET` | `/dashboard/summary` | Headcount, high-risk counts, average engagement |
| `GET` | `/dashboard/attrition-by-department` | Departmental attrition risk breakdown |
| `GET` | `/dashboard/skill-gaps` | Organization-wide critical skill gaps |
| `GET` | `/dashboard/recommendations` | List of tailored upskilling recommendations |
| `GET` | `/employees/{employee_id}` | 360-degree employee intelligence dossier |
| `GET` | `/skills/employee/{employee_id}` | Individual employee skill deficits |

---

## 🛡️ Monitoring & Retraining Policy

The platform includes automated continuous monitoring (`monitoring/drift_detector.py` and `monitoring/retraining_policy.py`):
```
IF Statistical Drift > 0.15
OR Production F1 Drops Below 0.65
OR 6 Months of New Data Collected
THEN Trigger Automated Retraining Pipeline
```

---

## 🌟 Future Roadmap
- Semantic course embedding with SentenceTransformers and vector search (FAISS/Milvus).
- Continuous real-time feedback loop integration with HRIS systems (Workday / BambooHR).
- LLM-powered AI retention interview script generator.
