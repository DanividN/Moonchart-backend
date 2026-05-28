from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, ConfigDict

class AudioAssetBase(BaseModel):
    s3_key: str
    asset_type: str
    status: str = "pending"

class AudioAssetCreate(AudioAssetBase):
    project_id: UUID | None = None

class AudioAssetResponse(AudioAssetBase):
    id: UUID
    project_id: UUID | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class ProcessingJobBase(BaseModel):
    job_type: str
    status: str = "pending"
    error_details: str | None = None
    result_data: dict | None = None
    options: dict = {}

class ProcessingJobCreate(ProcessingJobBase):
    project_id: UUID | None = None
    task_id_uuid: UUID | None = None

class ProcessingJobResponse(ProcessingJobBase):
    id: UUID
    project_id: UUID | None = None
    task_id_uuid: UUID | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
