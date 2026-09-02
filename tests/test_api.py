import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_dashboard_summary_endpoint():
    response = client.get("/dashboard/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_employees" in data
    assert "high_risk_employees" in data
    assert "average_engagement" in data

def test_attrition_by_department():
    response = client.get("/dashboard/attrition-by-department")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_skill_gaps_endpoint():
    response = client.get("/dashboard/skill-gaps")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_employee_dossier_found():
    response = client.get("/employees/1")
    assert response.status_code == 200
    data = response.json()
    assert data["EmployeeID"] == 1
    assert "AttritionProbability" in data
    assert "SkillGaps" in data

def test_employee_dossier_not_found():
    response = client.get("/employees/999999")
    assert response.status_code == 404

def test_predict_attrition_api():
    payload = {
        "EmployeeID": 888,
        "Age": 32,
        "Department": "Research & Development",
        "JobRole": "Research Scientist",
        "MonthlyIncome": 7000.0,
        "OverTime": "No"
    }
    response = client.post("/predict/attrition", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["EmployeeID"] == 888
    assert "AttritionProbability" in data
    assert data["RiskLevel"] in ["LOW", "MEDIUM", "HIGH"]
