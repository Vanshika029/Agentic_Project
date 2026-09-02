import pytest
from app.ml.predictor import AttritionPredictor
from app.ml.model_loader import ModelLoader

def test_model_loader_loads_pipeline():
    pipeline = ModelLoader.get_pipeline()
    assert pipeline is not None
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps

def test_attrition_prediction_returns_valid_output():
    predictor = AttritionPredictor()
    sample_emp = {
        "EmployeeID": 777,
        "Age": 28,
        "BusinessTravel": "Travel_Frequently",
        "Department": "Sales",
        "DistanceFromHome": 20,
        "Education": 3,
        "EducationField": "Marketing",
        "EnvironmentSatisfaction": 1,
        "Gender": "Male",
        "JobInvolvement": 2,
        "JobLevel": 1,
        "JobRole": "Sales Representative",
        "JobSatisfaction": 1,
        "MaritalStatus": "Single",
        "MonthlyIncome": 2500.0,
        "NumCompaniesWorked": 3,
        "OverTime": "Yes",
        "PercentSalaryHike": 11,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 1,
        "StockOptionLevel": 0,
        "TotalWorkingYears": 4,
        "TrainingTimesLastYear": 1,
        "WorkLifeBalance": 1,
        "YearsAtCompany": 2,
        "YearsInCurrentRole": 1,
        "YearsSinceLastPromotion": 0,
        "YearsWithCurrManager": 1
    }
    result = predictor.predict_single(sample_emp)
    
    assert "AttritionProbability" in result
    assert 0.0 <= result["AttritionProbability"] <= 1.0
    assert result["RiskLevel"] in ["LOW", "MEDIUM", "HIGH"]
    assert isinstance(result["TopRiskDrivers"], list)
    assert len(result["TopRiskDrivers"]) > 0
