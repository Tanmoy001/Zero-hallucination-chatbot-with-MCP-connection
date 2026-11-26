import asyncio
import sys
import threading

import streamlit as st
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

from config import DB_URI

# psycopg async does NOT work with the default Proactor loop on Windows.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class AsyncRunner:
    """Runs a persistent asyncio event loop in a daemon thread for Streamlit."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro, timeout: float | None = None):
        """Submit a coroutine to the background loop and block until done."""
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result(timeout=timeout)


async def _init_postgres():
    pool = AsyncConnectionPool(
        conninfo=DB_URI,
        min_size=1,
        max_size=10,
        open=False,
        check=AsyncConnectionPool.check_connection,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    # wait=True -> fail fast with a clear error if Postgres is unreachable
    await pool.open(wait=True, timeout=10)

    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()

    store = AsyncPostgresStore(pool)
    await store.setup()
    return pool, checkpointer, store


# Separate cached runner: if Postgres init fails, we do NOT create a new thread+loop on every rerun.
@st.cache_resource(show_spinner=False)
def get_runner() -> AsyncRunner:
    return AsyncRunner()


@st.cache_resource(show_spinner="Connecting to Postgres…")
def get_runtime():
    """Initialize the Postgres pool, checkpointer, and LTM store once."""
    runner = get_runner()
    try:
        pool, checkpointer, memory_store = runner.run(_init_postgres(), timeout=30)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot connect to Postgres at POSTGRES_URI ({exc!r}). "
            "Is the container running and is the port correct?"
        ) from exc
    return runner, pool, checkpointer, memory_store