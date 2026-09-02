from fastapi import APIRouter, HTTPException, status
from app.validation.employee_schema import EmployeePredictionInput, EmployeePredictionResponse
from app.services.attrition_service import AttritionService

router = APIRouter(prefix="/predict", tags=["Attrition ML"])
attrition_service = AttritionService()

@router.post("/attrition", response_model=EmployeePredictionResponse, status_code=status.HTTP_200_OK)
def predict_attrition(payload: EmployeePredictionInput):
    """Executes attrition prediction, risk tiering, and SHAP decision drivers for an employee."""
    try:
        data = payload.model_dump()
        result = attrition_service.predict(data)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )
