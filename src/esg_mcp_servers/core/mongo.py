"""MongoDB + GridFS singleton connection."""
import logging

import gridfs
from pymongo import MongoClient
from pymongo.database import Database

from esg_mcp_servers.settings import settings

logger = logging.getLogger(__name__)

_client: MongoClient | None = None
_db: Database | None = None
_fs: gridfs.GridFS | None = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(settings.MONGODB_URI)
        logger.info("MongoDB client initialised")
    return _client


def get_db() -> Database:
    global _db
    if _db is None:
        _db = get_client()[settings.MONGODB_DB]
    return _db


def get_gridfs() -> gridfs.GridFS:
    global _fs
    if _fs is None:
        _fs = gridfs.GridFS(get_db())
    return _fs


def close():
    global _client, _db, _fs
    if _client:
        _client.close()
        _client = None
        _db = None
        _fs = None
        logger.info("MongoDB connection closed")
