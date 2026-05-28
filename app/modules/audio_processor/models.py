from datetime import datetime, timezone
import uuid
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.core.database import Base

class AudioAsset(Base):
    __tablename__ = "audio_assets"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("projects.id"), index=True, nullable=True)
    s3_key: Mapped[str] = mapped_column(unique=True)
    asset_type: Mapped[str] # 'master_track', 'stem', 'midi_export'
    status: Mapped[str] = mapped_column(default="pending") # 'pending', 'processing', 'ready', 'failed'
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("projects.id"), index=True, nullable=True)
    task_id_uuid: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True) # Celery task id
    job_type: Mapped[str] # 'bpm_detection', 'source_separation', 'midi_generation'
    status: Mapped[str] = mapped_column(default="pending") # 'pending', 'processing', 'completed', 'failed'
    error_details: Mapped[str | None] = mapped_column(default=None)
    result_data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
