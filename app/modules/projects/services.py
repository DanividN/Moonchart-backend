from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.modules.projects.models import Project
from app.modules.projects.schemas import ProjectCreate

async def get_projects_by_user(db: AsyncSession, user_id: UUID, skip: int = 0, limit: int = 100) -> list[Project]:
    result = await db.execute(select(Project).where(Project.owner_id == user_id).offset(skip).limit(limit))
    return list(result.scalars().all())

async def create_project(db: AsyncSession, project_in: ProjectCreate, user_id: UUID) -> Project:
    db_project = Project(
        owner_id=user_id,
        title=project_in.title,
        artist=project_in.artist,
        bpm_base=project_in.bpm_base,
        visibility=project_in.visibility
    )
    db.add(db_project)
    await db.commit()
    await db.refresh(db_project)
    return db_project

async def get_project(db: AsyncSession, project_id: UUID) -> Project | None:
    result = await db.execute(select(Project).where(Project.id == project_id))
    return result.scalars().first()
