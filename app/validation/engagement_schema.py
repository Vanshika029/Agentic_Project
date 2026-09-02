from pydantic import BaseModel, Field

class EngagementValidationInput(BaseModel):
    EmployeeID: int
    Department: str
    JobRole: str
    EngagementScore: int = Field(..., ge=0, le=100, description="Score between 0 and 100")
    PerformanceRating: int = Field(default=3, ge=1, le=5)
    WorkLifeBalanceScore: int = Field(default=3, ge=1, le=5)
