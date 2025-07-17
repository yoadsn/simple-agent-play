from contextlib import asynccontextmanager
from typing import AsyncGenerator

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

LG_MEM_DATABASE_URL = "postgresql://localhost:5432/agent_lg_mem"


@asynccontextmanager
async def get_checkpointer() -> AsyncGenerator[AsyncPostgresSaver, None]:
    async with AsyncPostgresSaver.from_conn_string(LG_MEM_DATABASE_URL) as cp:
        yield cp


async def setup_checkpointer_db():
    async with get_checkpointer() as cp:
        await cp.setup()
