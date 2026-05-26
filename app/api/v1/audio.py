from uuid import UUID
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.api.dependencies import get_current_user
from app.modules.users.models import User
from app.modules.projects.models import Project
from app.modules.audio_processor import schemas
from app.modules.audio_processor.models import AudioAsset, ProcessingJob
from app.modules.audio_processor.tasks import process_audio_asset

router = APIRouter()

@router.post("/assets", response_model=schemas.AudioAssetResponse)
async def create_audio_asset(
    asset_in: schemas.AudioAssetCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)]
):
    # Verify project exists and belongs to user
    project = await db.get(Project, asset_in.project_id)
    if not project or project.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to add assets to this project")
        
    db_asset = AudioAsset(
        project_id=asset_in.project_id,
        s3_key=asset_in.s3_key,
        asset_type=asset_in.asset_type,
        status="pending"
    )
    db.add(db_asset)
    await db.commit()
    await db.refresh(db_asset)
    return db_asset

@router.post("/process", response_model=schemas.ProcessingJobResponse)
async def trigger_processing(
    job_in: schemas.ProcessingJobCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)]
):
    # Verify project
    project = await db.get(Project, job_in.project_id)
    if not project or project.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    # Create job record
    db_job = ProcessingJob(
        project_id=job_in.project_id,
        job_type=job_in.job_type,
        status="processing"
    )
    db.add(db_job)
    await db.commit()
    await db.refresh(db_job)
    
    # Trigger Celery Task
    # Assume we pass a dummy s3_key for this skeleton
    task = process_audio_asset.delay(str(db_job.id), str(job_in.project_id), "dummy_s3_key", job_in.job_type)
    
    # Update job with task id
    db_job.task_id_uuid = UUID(task.id) if task.id else None
    await db.commit()
    await db.refresh(db_job)
    
    return db_job

from fastapi.responses import FileResponse
import os

@router.get("/download/midi/{job_id}")
async def download_midi(
    job_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """Permite la descarga directa del archivo MIDI cuantizado generado."""
    job = await db.get(ProcessingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job no encontrado")
        
    # Verificar propiedad del proyecto
    project = await db.get(Project, job.project_id)
    if not project or project.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="No autorizado")
        
    midi_filename = f"chart_{job_id}.mid"
    local_path = os.path.join("storage", "output", str(job_id), midi_filename)
    
    if not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail="El archivo MIDI aún no ha sido generado o no existe localmente")
        
    return FileResponse(
        path=local_path,
        filename=midi_filename,
        media_type="audio/midi"
    )

@router.get("/download/stem/{job_id}/{stem_name}")
async def download_stem(
    job_id: UUID,
    stem_name: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """Permite la descarga interactiva de cada pista o stem (vocals, drums, bass, guitar, song)."""
    if stem_name not in ["vocals", "drums", "bass", "guitar", "song"]:
        raise HTTPException(status_code=400, detail="Nombre de stem no válido. Use: vocals, drums, bass, guitar o song")
        
    job = await db.get(ProcessingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job no encontrado")
        
    project = await db.get(Project, job.project_id)
    if not project or project.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="No autorizado")
        
    ext = "ogg" if stem_name == "song" else "wav"
    media_type = "audio/ogg" if stem_name == "song" else "audio/wav"
    local_path = os.path.join("storage", "output", str(job_id), f"{stem_name}.{ext}")
    
    if not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail=f"El recurso '{stem_name}' no existe o no ha sido procesado aún")
        
    return FileResponse(
        path=local_path,
        filename=f"{stem_name}_{job_id}.{ext}",
        media_type=media_type
    )

from fastapi import UploadFile, File
import shutil

@router.post("/upload", response_model=schemas.AudioAssetResponse)
async def upload_audio_file(
    project_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    file: UploadFile = File(...)
):
    """Permite al frontend subir un archivo de audio real directamente al servidor."""
    # Verificar propiedad del proyecto
    project = await db.get(Project, project_id)
    if not project or project.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="No autorizado")
        
    local_dir = os.path.join("storage", "input")
    os.makedirs(local_dir, exist_ok=True)
    
    # Crear un nombre de archivo seguro
    safe_filename = f"{project_id}_{file.filename}"
    local_path = os.path.join(local_dir, "input.wav") # Reutilizar/guardar temporal
    
    # Escribir el archivo recibido localmente
    with open(local_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Crear asset en la base de datos
    db_asset = AudioAsset(
        project_id=project_id,
        s3_key=f"projects/{project_id}/input/{safe_filename}",
        asset_type="master_track",
        status="ready"
    )
    db.add(db_asset)
    await db.commit()
    await db.refresh(db_asset)
    
    logger.info("Uploaded master track successfully", project_id=project_id, filename=file.filename)
    return db_asset

@router.get("/jobs/{job_id}", response_model=schemas.ProcessingJobResponse)
async def get_job_status(
    job_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """Permite al frontend consultar el progreso del análisis de forma interactiva (polling)."""
    job = await db.get(ProcessingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job no encontrado")
        
    # Verificar acceso al proyecto
    project = await db.get(Project, job.project_id)
    if not project or project.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="No autorizado")
        
    return job


