from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from app.core.config import settings
from app.api.v1 import users, projects, audio

logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup actions (e.g., connect to DB if needed globally, init redis connection pool)
    await logger.ainfo("Starting up Music Charts API...")
    yield
    # Teardown actions
    await logger.ainfo("Shutting down Music Charts API...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

from fastapi.responses import JSONResponse
from fastapi.requests import Request
import traceback

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    with open("error.log", "a") as f:
        f.write(f"Exception on {request.url.path}:\n")
        traceback.print_exc(file=f)
    return JSONResponse(
        status_code=500,
        content={"message": str(exc)},
    )


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production this should be restricted
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}

app.include_router(users.router, prefix=f"{settings.API_V1_STR}/users", tags=["Users"])
app.include_router(projects.router, prefix=f"{settings.API_V1_STR}/projects", tags=["Projects"])
app.include_router(audio.router, prefix=f"{settings.API_V1_STR}/audio", tags=["Audio"])
