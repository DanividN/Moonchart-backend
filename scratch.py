import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

async def get_jobs():
    engine = create_async_engine('postgresql+asyncpg://chartuser:chartpassword@127.0.0.1:5432/chartdb')
    async with engine.connect() as conn:
        result = await conn.execute(text('SELECT id, status, error_details FROM processing_jobs ORDER BY created_at DESC LIMIT 5'))
        print(result.fetchall())

asyncio.run(get_jobs())
