from datetime import datetime, timezone
import uuid
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.core.database import Base

class Project(Base):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str]
    artist: Mapped[str]
    bpm_base: Mapped[float | None] = mapped_column(default=None)
    visibility: Mapped[str] = mapped_column(default="private") # 'private' or 'public'
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

class InstrumentTrack(Base):
    __tablename__ = "instruments_tracks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id"), index=True)
    instrument_type: Mapped[str] # e.g., 'guitar', 'bass', 'drums'
    difficulty: Mapped[str] # e.g., 'expert', 'hard'
    track_metadata: Mapped[dict] = mapped_column(JSONB, default=dict)
