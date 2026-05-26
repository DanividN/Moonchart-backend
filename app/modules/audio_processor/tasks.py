import os
import time
import asyncio
from uuid import UUID
from app.worker.celery_app import celery_app
from app.core.database import AsyncSessionLocal
from app.modules.audio_processor.models import ProcessingJob, AudioAsset
from app.modules.audio_processor.processor import AudioProcessor
import structlog

logger = structlog.get_logger()

async def update_job_status(job_id: str, status: str, error_details: str | None = None):
    """Updates the status of a ProcessingJob asynchronously."""
    async with AsyncSessionLocal() as db:
        job = await db.get(ProcessingJob, UUID(job_id))
        if job:
            job.status = status
            if error_details:
                job.error_details = error_details
            await db.commit()
            logger.info("Updated job status in DB", job_id=job_id, status=status)

async def save_midi_asset(project_id: str, s3_key: str):
    """Saves the generated MIDI file metadata asynchronously."""
    async with AsyncSessionLocal() as db:
        asset = AudioAsset(
            project_id=UUID(project_id),
            s3_key=s3_key,
            asset_type="midi_export",
            status="ready"
        )
        db.add(asset)
        await db.commit()
        logger.info("Saved MIDI asset metadata in DB", project_id=project_id, s3_key=s3_key)

@celery_app.task(bind=True, name="process_audio_asset")
def process_audio_asset(self, job_id: str, project_id: str, s3_key: str, job_type: str):
    logger.info("Starting audio processing task", job_id=job_id, project_id=project_id, job_type=job_type)
    
    # Set to processing in database
    asyncio.run(update_job_status(job_id, "processing"))
    
    try:
        # 1. Download file or use fallback mock audio
        logger.info("Downloading/Resolving audio asset", s3_key=s3_key)
        
        # Local paths for processing
        local_input_dir = "storage/input"
        local_output_dir = f"storage/output/{job_id}"
        os.makedirs(local_input_dir, exist_ok=True)
        os.makedirs(local_output_dir, exist_ok=True)
        
        # Mock a generated waveform or use a real sample if it exists.
        # Create a dummy silent wav if no file is provided to allow safe processing execution
        input_audio_path = os.path.join(local_input_dir, "input.wav")
        if not os.path.exists(input_audio_path):
            # Write a 5-second sine wave as fallback for testing
            import numpy as np
            import soundfile as sf
            sr = 44100
            t = np.linspace(0, 5, sr * 5)
            # Create a 440 Hz synth sound for rítmica
            y = np.sin(2 * np.pi * 440 * t) * 0.5
            sf.write(input_audio_path, y, sr)
            logger.info("Synthesized test audio waveform for analysis")
            
        # Parse target instrument from job_type (e.g. "midi_generation:guitar")
        target_instrument = "guitar"
        if ":" in job_type:
            _, target_instrument = job_type.split(":", 1)
        target_instrument = target_instrument.lower()
        if target_instrument not in ["guitar", "bass", "drums", "vocals"]:
            target_instrument = "guitar"

        # 2. Run AudioProcessor
        processor = AudioProcessor(sample_rate=44100)
        
        # Analyze BPM and beats of overall master track
        y_master, sr_master = processor.load_audio(input_audio_path)
        tempo_data = processor.analyze_tempo(y_master)
        
        # Separate Stems
        stems = processor.separate_stems_mock(input_audio_path, local_output_dir)
        
        # Load the specific stem file for the target instrument
        stem_path = stems.get(target_instrument, input_audio_path)
        if os.path.exists(stem_path):
            logger.info("Loading isolated stem for precision onset detection", instrument=target_instrument, path=stem_path)
            y_stem, sr_stem = processor.load_audio(stem_path)
        else:
            logger.warn("Stem file not found, falling back to master track", instrument=target_instrument)
            y_stem, sr_stem = y_master, sr_master
            
        # Detect precise onsets for the target instrument on its isolated stem
        onsets = processor.detect_notes_and_onsets(y_stem, instrument=target_instrument)
        
        # Standardize track name for Clone Hero / Rock Band MIDI
        midi_track_map = {
            "guitar": "PART GUITAR",
            "bass": "PART BASS",
            "drums": "PART DRUMS",
            "vocals": "PART VOCALS"
        }
        track_name = midi_track_map.get(target_instrument, "PART GUITAR")
        
        track_onsets = {
            track_name: onsets
        }
        
        # 3. Generate MIDI Chart
        midi_filename = f"chart_{job_id}.mid"
        local_midi_path = os.path.join(local_output_dir, midi_filename)
        processor.generate_midi_chart(tempo_data, track_onsets, local_midi_path)
        
        # 4. Upload results (mock/save path as S3 key)
        midi_s3_key = f"projects/{project_id}/charts/{midi_filename}"
        
        # Save to DB asynchronously
        asyncio.run(save_midi_asset(project_id, midi_s3_key))
        asyncio.run(update_job_status(job_id, "completed"))
        
        logger.info("Audio processing task completed successfully", job_id=job_id)
        return {
            "status": "completed",
            "job_id": job_id,
            "project_id": project_id,
            "midi_s3_key": midi_s3_key,
            "bpm": tempo_data["bpm"]
        }
    except Exception as e:
        logger.error("Audio processing task failed", job_id=job_id, error=str(e))
        asyncio.run(update_job_status(job_id, "failed", error_details=str(e)))
        self.update_state(state="FAILURE", meta={"exc_type": type(e).__name__, "exc_message": str(e)})
        raise

