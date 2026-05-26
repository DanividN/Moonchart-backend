from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, ConfigDict
from typing import Any

class InstrumentTrackBase(BaseModel):
    instrument_type: str
    difficulty: str
    track_metadata: dict[str, Any] = {}

class InstrumentTrackCreate(InstrumentTrackBase):
    pass

class InstrumentTrackResponse(InstrumentTrackBase):
    id: UUID
    project_id: UUID

    model_config = ConfigDict(from_attributes=True)

class ProjectBase(BaseModel):
    title: str
    artist: str
    bpm_base: float
    visibility: str = "private"

class ProjectCreate(ProjectBase):
    pass

class ProjectResponse(ProjectBase):
    id: UUID
    owner_id: UUID
    created_at: datetime
    updated_at: datetime
    tracks: list[InstrumentTrackResponse] = []

    model_config = ConfigDict(from_attributes=True)
