import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from fastapi import UploadFile
from starlette.datastructures import Headers
import tempfile
import os

from app.api.v1.audio import upload_audio_file
from app.core.config import settings

async def main():
    engine = create_async_engine(settings.SQLALCHEMY_DATABASE_URI)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b'test')
        tmp_path = tmp.name
        
    try:
        with open(tmp_path, 'rb') as f:
            upload_file = UploadFile(filename='test.wav', file=f, headers=Headers({'content-type': 'audio/wav'}))
            async with TestingSessionLocal() as db:
                await upload_audio_file(db=db, project_id=None, file=upload_file)
    except Exception as e:
        import traceback
        traceback.print_exc()
        
    os.remove(tmp_path)

asyncio.run(main())
