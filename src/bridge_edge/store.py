"""Local SQLite history; a separate connection per request/thread."""
import json
import sqlite3
from contextlib import contextmanager

from .paths import workspace


@contextmanager
def _connect():
    connection = sqlite3.connect(workspace() / "history.sqlite", timeout=15)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            connection.execute("CREATE TABLE IF NOT EXISTS predictions (id TEXT PRIMARY KEY, created_utc TEXT NOT NULL, model TEXT NOT NULL, source_filename TEXT, finding_count INTEGER NOT NULL, latency_ms REAL NOT NULL, result_json TEXT NOT NULL)")
            yield connection
    finally:
        connection.close()


def record_prediction(result):
    with _connect() as connection:
        connection.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?,?)",
                           (result["id"], result["created_utc"], result["model"], result["source_filename"],
                            result["finding_count"], result["total_latency_ms"], json.dumps(result, ensure_ascii=False)))


def list_predictions(limit=30, offset=0):
    if not 1 <= limit <= 100 or offset < 0:
        raise ValueError("Invalid history pagination")
    with _connect() as connection:
        rows = connection.execute("SELECT id,created_utc,model,source_filename,finding_count,latency_ms FROM predictions ORDER BY created_utc DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
    return [dict(row) for row in rows]


def get_prediction(identifier):
    with _connect() as connection:
        row = connection.execute("SELECT result_json FROM predictions WHERE id=?", (identifier,)).fetchone()
    return json.loads(row[0]) if row else None
