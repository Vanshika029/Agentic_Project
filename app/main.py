from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.utils.config import PROJECT_NAME
from app.utils.logger import logger
from app.ml.model_loader import ModelLoader
from app.api import attrition, dashboard, skills

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Enterprise HR AI Platform...")
    try:
        ModelLoader.get_pipeline()
        logger.info("Machine Learning Pipeline & Model Registry loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading ML Pipeline: {e}")
    yield
    logger.info("Enterprise HR AI Platform shutdown complete.")

app = FastAPI(
    title=PROJECT_NAME,
    description="Agentic Enterprise HR Intelligence Platform predicting attrition, tracking engagement, evaluating skill gaps, and recommending upskilling courses.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for dashboard and external integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(attrition.router)
app.include_router(dashboard.router)
app.include_router(skills.router)

@app.get("/", tags=["System"])
def root():
    return {
        "status": "online",
        "system": PROJECT_NAME,
        "version": "1.0.0",
        "docs_url": "/docs"
    }

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "healthy"}
