import pytest
from pydantic import ValidationError
from app.validation.employee_schema import EmployeePredictionInput
from app.validation.engagement_schema import EngagementValidationInput

def test_valid_employee_schema():
    valid_payload = {
        "EmployeeID": 101,
        "Age": 35,
        "Department": "Research & Development",
        "JobRole": "Research Scientist",
        "MonthlyIncome": 6500.0,
        "OverTime": "Yes",
        "TotalWorkingYears": 10,
        "YearsAtCompany": 5,
        "JobSatisfaction": 3
    }
    obj = EmployeePredictionInput(**valid_payload)
    assert obj.EmployeeID == 101
    assert obj.Age == 35
    assert obj.OverTime == "Yes"

def test_invalid_age_raises_validation_error():
    invalid_payload = {
        "EmployeeID": 102,
        "Age": 15,  # Below minimum 18
        "Department": "Sales",
        "JobRole": "Sales Executive",
        "MonthlyIncome": 5000.0,
        "OverTime": "No"
    }
    with pytest.raises(ValidationError):
        EmployeePredictionInput(**invalid_payload)

def test_invalid_overtime_raises_validation_error():
    invalid_payload = {
        "EmployeeID": 103,
        "Age": 30,
        "Department": "Sales",
        "JobRole": "Sales Executive",
        "MonthlyIncome": 5000.0,
        "OverTime": "Maybe"  # Must be 'Yes' or 'No'
    }
    with pytest.raises(ValidationError):
        EmployeePredictionInput(**invalid_payload)

def test_invalid_engagement_score_raises_error():
    with pytest.raises(ValidationError):
        EngagementValidationInput(
            EmployeeID=101,
            Department="IT",
            JobRole="Engineer",
            EngagementScore=150  # Must be 0-100
        )
