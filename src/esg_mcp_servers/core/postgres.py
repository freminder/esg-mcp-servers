"""PostgreSQL connection pool using psycopg2."""
import logging
import os
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool
from pgvector.psycopg2 import register_vector

from esg_mcp_servers.settings import settings

logger = logging.getLogger(__name__)

_pool: pool.ThreadedConnectionPool | None = None


def get_pool() -> pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=50,
            dsn=settings.POSTGRES_DSN,
        )
        # Register pgvector type on a throwaway connection to warm up
        conn = _pool.getconn()
        try:
            register_vector(conn)
        finally:
            _pool.putconn(conn)
        logger.info("PostgreSQL connection pool initialised")
    return _pool


def _get_healthy_conn(p: pool.ThreadedConnectionPool):
    """Get a connection from the pool, replacing stale ones."""
    for _ in range(3):
        conn = p.getconn()
        if conn.closed:
            logger.debug("Discarding closed pooled connection")
            p.putconn(conn, close=True)
            continue
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
            conn.commit()
            return conn
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            logger.debug("Discarding stale pooled connection")
            p.putconn(conn, close=True)
            continue
    return p.getconn()


@contextmanager
def get_connection():
    """Context manager: borrow a connection, auto-return on exit."""
    p = get_pool()
    conn = _get_healthy_conn(p)
    try:
        register_vector(conn)
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            p.putconn(conn)
        except Exception:
            pass


def run_migrations():
    """Apply SQL migration files in order."""
    migration_dir = os.path.join(
        os.path.dirname(__file__), "..", "migrations",
    )
    migration_dir = os.path.normpath(migration_dir)
    files = sorted(f for f in os.listdir(migration_dir) if f.endswith(".sql"))

    with get_connection() as conn:
        cur = conn.cursor()
        for filename in files:
            path = os.path.join(migration_dir, filename)
            with open(path) as f:
                sql = f.read()
            logger.info(f"Running migration: {filename}")
            cur.execute(sql)
        cur.close()
    logger.info("All migrations applied successfully")


def close_pool():
    global _pool
    if _pool:
        _pool.closeall()
        _pool = None
        logger.info("PostgreSQL connection pool closed")
