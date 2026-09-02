from pydantic import BaseModel, Field, field_validator
from typing import Optional, List

class EmployeePredictionInput(BaseModel):
    EmployeeID: Optional[int] = Field(default=9999, description="Unique Employee Identifier")
    Age: int = Field(..., ge=18, le=100, description="Age between 18 and 100")
    BusinessTravel: str = Field(default="Travel_Rarely", description="Travel frequency")
    DailyRate: Optional[int] = Field(default=800, ge=100, le=2000)
    Department: str = Field(..., description="Department: Sales, Research & Development, Human Resources")
    DistanceFromHome: int = Field(default=5, ge=1, le=50)
    Education: int = Field(default=3, ge=1, le=5)
    EducationField: str = Field(default="Life Sciences")
    EnvironmentSatisfaction: int = Field(default=3, ge=1, le=4)
    Gender: str = Field(default="Male")
    HourlyRate: Optional[int] = Field(default=65, ge=20, le=150)
    JobInvolvement: int = Field(default=3, ge=1, le=4)
    JobLevel: int = Field(default=2, ge=1, le=5)
    JobRole: str = Field(..., description="Target enterprise Job Role")
    JobSatisfaction: int = Field(default=3, ge=1, le=4)
    MaritalStatus: str = Field(default="Single")
    MonthlyIncome: float = Field(..., gt=0, description="Monthly income in USD")
    MonthlyRate: Optional[int] = Field(default=15000, ge=1000, le=40000)
    NumCompaniesWorked: int = Field(default=2, ge=0, le=15)
    OverTime: str = Field(..., description="'Yes' or 'No'")
    PercentSalaryHike: int = Field(default=15, ge=0, le=50)
    PerformanceRating: int = Field(default=3, ge=1, le=4)
    RelationshipSatisfaction: int = Field(default=3, ge=1, le=4)
    StockOptionLevel: int = Field(default=1, ge=0, le=3)
    TotalWorkingYears: int = Field(default=10, ge=0, le=50)
    TrainingTimesLastYear: int = Field(default=2, ge=0, le=10)
    WorkLifeBalance: int = Field(default=3, ge=1, le=4)
    YearsAtCompany: int = Field(default=5, ge=0, le=45)
    YearsInCurrentRole: int = Field(default=3, ge=0, le=30)
    YearsSinceLastPromotion: int = Field(default=1, ge=0, le=30)
    YearsWithCurrManager: int = Field(default=3, ge=0, le=30)

    @field_validator('OverTime')
    @classmethod
    def validate_overtime(cls, v: str) -> str:
        if v not in ('Yes', 'No'):
            raise ValueError("OverTime must be either 'Yes' or 'No'")
        return v

    @field_validator('Department')
    @classmethod
    def validate_department(cls, v: str) -> str:
        valid_depts = {'Sales', 'Research & Development', 'Human Resources'}
        if v not in valid_depts:
            raise ValueError(f"Department must be one of {valid_depts}")
        return v


class EmployeePredictionResponse(BaseModel):
    EmployeeID: int
    AttritionProbability: float
    RiskLevel: str
    TopRiskDrivers: List[str]
    ModelVersion: str
    Timestamp: str
