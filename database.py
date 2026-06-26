"""
database.py — J.A.R.V.I.S Persistence Layer
SQLite database replacing all JSON files. Handles:
  - Chat history (all conversations)
  - Projects (full CRUD)
  - Curiosity insights index
  - System metrics snapshots
  - Documents registry
  - Training runs history
"""

import sqlite3
import json
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import contextmanager


DB_PATH = "data/jarvis.db"


class JarvisDB:
    """Thread-safe SQLite wrapper for J.A.R.V.I.S persistence."""

    def __init__(self, path: str = DB_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._init_schema()
        print(f"[DB] SQLite initialized -> {path}")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")   # write-ahead log for concurrency
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._conn() as c:
            c.executescript("""
                -- Chat history
                CREATE TABLE IF NOT EXISTS messages (
                    id          TEXT PRIMARY KEY,
                    role        TEXT NOT NULL,          -- 'user' | 'jarvis'
                    content     TEXT NOT NULL,
                    intent      TEXT,
                    sources     TEXT,                   -- JSON array
                    ts          REAL NOT NULL,
                    session_id  TEXT NOT NULL
                );

                -- Projects
                CREATE TABLE IF NOT EXISTS projects (
                    id          TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    type        TEXT,
                    priority    TEXT DEFAULT 'BETA',
                    description TEXT,
                    tags        TEXT,                   -- JSON array
                    status      TEXT DEFAULT 'ATIVO',
                    files       TEXT,                   -- JSON array of generated files
                    created_at  REAL NOT NULL,
                    updated_at  REAL NOT NULL
                );

                -- Curiosity insights
                CREATE TABLE IF NOT EXISTS insights (
                    id               TEXT PRIMARY KEY,
                    source           TEXT NOT NULL,
                    chunk_text       TEXT NOT NULL,
                    summary          TEXT NOT NULL,
                    tags             TEXT,              -- JSON array
                    curiosity_score  REAL,
                    novelty_score    REAL,
                    entropy_score    REAL,
                    surprise_score   REAL,
                    connections      TEXT,              -- JSON array of insight IDs
                    times_surfaced   INTEGER DEFAULT 0,
                    is_new           INTEGER DEFAULT 1,
                    ts               REAL NOT NULL
                );

                -- System metrics snapshots (ring buffer)
                CREATE TABLE IF NOT EXISTS metrics (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts       REAL NOT NULL,
                    cpu_pct  REAL,
                    ram_pct  REAL,
                    disk_pct REAL,
                    temp     REAL,
                    net_sent REAL,
                    snapshot TEXT                       -- full JSON
                );

                -- Documents registry
                CREATE TABLE IF NOT EXISTS documents (
                    id        TEXT PRIMARY KEY,
                    filename  TEXT NOT NULL,
                    path      TEXT,
                    pages     INTEGER,
                    words     INTEGER,
                    tables    INTEGER,
                    images    INTEGER,
                    language  TEXT,
                    indexed   INTEGER DEFAULT 0,
                    corpus    TEXT,                     -- extracted training corpus
                    ts        REAL NOT NULL
                );

                -- Training runs
                CREATE TABLE IF NOT EXISTS training_runs (
                    id          TEXT PRIMARY KEY,
                    started_at  REAL NOT NULL,
                    finished_at REAL,
                    status      TEXT DEFAULT 'running', -- 'running'|'done'|'error'
                    steps       INTEGER DEFAULT 0,
                    final_loss  REAL,
                    vocab_size  INTEGER,
                    history     TEXT                   -- JSON array of step logs
                );

                -- Curiosity cycle log
                CREATE TABLE IF NOT EXISTS curiosity_cycles (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          REAL NOT NULL,
                    cycle_num   INTEGER,
                    docs_scanned INTEGER,
                    insights_found INTEGER,
                    top_score   REAL
                );

                -- Indices
                CREATE INDEX IF NOT EXISTS idx_messages_ts      ON messages(ts DESC);
                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
                CREATE INDEX IF NOT EXISTS idx_insights_score   ON insights(curiosity_score DESC);
                CREATE INDEX IF NOT EXISTS idx_metrics_ts       ON metrics(ts DESC);
                CREATE INDEX IF NOT EXISTS idx_documents_ts     ON documents(ts DESC);
            """)

    # ─── Messages ─────────────────────────────────────────────────────────────

    def save_message(self, role: str, content: str, session_id: str,
                     intent: str = None, sources: List = None) -> str:
        mid = str(uuid.uuid4())
        with self._conn() as c:
            c.execute(
                "INSERT INTO messages VALUES (?,?,?,?,?,?,?)",
                (mid, role, content, intent,
                 json.dumps(sources or []), time.time(), session_id)
            )
        return mid

    def get_history(self, session_id: str = None, limit: int = 50) -> List[Dict]:
        with self._conn() as c:
            if session_id:
                rows = c.execute(
                    "SELECT * FROM messages WHERE session_id=? ORDER BY ts DESC LIMIT ?",
                    (session_id, limit)
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM messages ORDER BY ts DESC LIMIT ?", (limit,)
                ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def get_sessions(self) -> List[Dict]:
        with self._conn() as c:
            rows = c.execute("""
                SELECT session_id,
                       COUNT(*) as msg_count,
                       MIN(ts) as started,
                       MAX(ts) as last_ts
                FROM messages
                GROUP BY session_id
                ORDER BY last_ts DESC
                LIMIT 20
            """).fetchall()
        return [dict(r) for r in rows]

    # ─── Projects ─────────────────────────────────────────────────────────────

    def save_project(self, name: str, type_: str, priority: str = "BETA",
                     description: str = "", tags: List = None,
                     files: List = None) -> Dict:
        pid = str(uuid.uuid4())[:8].upper()
        now = time.time()
        with self._conn() as c:
            c.execute(
                "INSERT INTO projects VALUES (?,?,?,?,?,?,?,?,?,?)",
                (pid, name, type_, priority, description,
                 json.dumps(tags or []), "ATIVO",
                 json.dumps(files or []), now, now)
            )
        return self.get_project(pid)

    def get_project(self, pid: str) -> Optional[Dict]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["tags"]  = json.loads(d["tags"]  or "[]")
        d["files"] = json.loads(d["files"] or "[]")
        return d

    def list_projects(self, status: str = None) -> List[Dict]:
        with self._conn() as c:
            if status:
                rows = c.execute(
                    "SELECT * FROM projects WHERE status=? ORDER BY created_at DESC", (status,)
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM projects ORDER BY created_at DESC"
                ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["tags"]  = json.loads(d["tags"]  or "[]")
            d["files"] = json.loads(d["files"] or "[]")
            result.append(d)
        return result

    def update_project(self, pid: str, **kwargs) -> Optional[Dict]:
        allowed = {"name","type","priority","description","tags","status","files"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return self.get_project(pid)
        updates["updated_at"] = time.time()
        for k in ("tags","files"):
            if k in updates and isinstance(updates[k], list):
                updates[k] = json.dumps(updates[k])
        cols = ", ".join(f"{k}=?" for k in updates)
        vals = list(updates.values()) + [pid]
        with self._conn() as c:
            c.execute(f"UPDATE projects SET {cols} WHERE id=?", vals)
        return self.get_project(pid)

    # ─── Insights ─────────────────────────────────────────────────────────────

    def save_insight(self, ins: Dict) -> str:
        with self._conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO insights
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (ins["id"], ins["source"], ins["chunk_text"][:800],
                 ins["summary"], json.dumps(ins.get("tags", [])),
                 ins.get("curiosity_score", 0), ins.get("novelty_score", 0),
                 ins.get("entropy_score", 0),   ins.get("surprise_score", 0),
                 json.dumps(ins.get("connections", [])),
                 ins.get("times_surfaced", 0),  int(ins.get("is_new", True)),
                 ins.get("timestamp", time.time()))
            )
        return ins["id"]

    def get_insights(self, limit: int = 50, tag: str = None,
                     min_score: float = 0.0) -> List[Dict]:
        with self._conn() as c:
            if tag:
                rows = c.execute(
                    """SELECT * FROM insights
                       WHERE curiosity_score >= ? AND tags LIKE ?
                       ORDER BY curiosity_score DESC LIMIT ?""",
                    (min_score, f'%{tag}%', limit)
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM insights WHERE curiosity_score >= ? ORDER BY curiosity_score DESC LIMIT ?",
                    (min_score, limit)
                ).fetchall()
        return [self._parse_insight(r) for r in rows]

    def search_insights(self, query: str, limit: int = 20) -> List[Dict]:
        q = f"%{query}%"
        with self._conn() as c:
            rows = c.execute(
                """SELECT * FROM insights
                   WHERE chunk_text LIKE ? OR summary LIKE ? OR tags LIKE ?
                   ORDER BY curiosity_score DESC LIMIT ?""",
                (q, q, q, limit)
            ).fetchall()
        return [self._parse_insight(r) for r in rows]

    def get_random_insight(self, min_score: float = 0.3) -> Optional[Dict]:
        with self._conn() as c:
            row = c.execute(
                """SELECT * FROM insights WHERE curiosity_score >= ?
                   ORDER BY times_surfaced ASC, RANDOM() LIMIT 1""",
                (min_score,)
            ).fetchone()
            if row:
                c.execute("UPDATE insights SET times_surfaced=times_surfaced+1, is_new=0 WHERE id=?",
                          (row["id"],))
        return self._parse_insight(row) if row else None

    def get_topics(self) -> Dict[str, int]:
        with self._conn() as c:
            rows = c.execute("SELECT tags FROM insights WHERE tags IS NOT NULL").fetchall()
        counts: Dict[str, int] = {}
        for row in rows:
            for tag in json.loads(row["tags"] or "[]"):
                counts[tag] = counts.get(tag, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    def _parse_insight(self, row) -> Dict:
        if not row:
            return {}
        d = dict(row)
        d["tags"]        = json.loads(d.get("tags", "[]")        or "[]")
        d["connections"] = json.loads(d.get("connections", "[]") or "[]")
        return d

    # ─── Metrics ──────────────────────────────────────────────────────────────

    def save_metric(self, snap: Dict):
        cpu  = snap.get("cpu", {})
        mem  = snap.get("memory", {})
        disk = snap.get("disk", {})
        net  = snap.get("network", {})
        with self._conn() as c:
            c.execute(
                "INSERT INTO metrics (ts,cpu_pct,ram_pct,disk_pct,temp,net_sent,snapshot) VALUES (?,?,?,?,?,?,?)",
                (time.time(), cpu.get("percent"), mem.get("percent"),
                 disk.get("percent"), cpu.get("temperature"),
                 net.get("bytes_sent_mb"),
                 json.dumps(snap, default=str))
            )
            # Ring buffer — keep last 2000 samples
            c.execute("DELETE FROM metrics WHERE id NOT IN (SELECT id FROM metrics ORDER BY ts DESC LIMIT 2000)")

    def get_metrics_history(self, minutes: int = 30) -> List[Dict]:
        since = time.time() - minutes * 60
        with self._conn() as c:
            rows = c.execute(
                "SELECT ts,cpu_pct,ram_pct,disk_pct,temp,net_sent FROM metrics WHERE ts > ? ORDER BY ts ASC",
                (since,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ─── Documents ────────────────────────────────────────────────────────────

    def save_document(self, filename: str, path: str, stats: Dict,
                      corpus: str = "") -> str:
        did = str(uuid.uuid4())[:8]
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (did, filename, path, stats.get("pages", 0),
                 stats.get("words", 0), stats.get("tables", 0),
                 stats.get("images", 0), stats.get("language", ""),
                 0, corpus[:50000], time.time())
            )
        return did

    def list_documents(self) -> List[Dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id,filename,path,pages,words,tables,images,language,indexed,ts FROM documents ORDER BY ts DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_indexed(self, doc_id: str):
        with self._conn() as c:
            c.execute("UPDATE documents SET indexed=1 WHERE id=?", (doc_id,))

    # ─── Training Runs ────────────────────────────────────────────────────────

    def start_training_run(self) -> str:
        rid = str(uuid.uuid4())[:8]
        with self._conn() as c:
            c.execute(
                "INSERT INTO training_runs (id,started_at,status,history) VALUES (?,?,?,?)",
                (rid, time.time(), "running", "[]")
            )
        return rid

    def update_training_run(self, rid: str, **kwargs):
        allowed = {"finished_at","status","steps","final_loss","vocab_size","history"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if "history" in updates and isinstance(updates["history"], list):
            updates["history"] = json.dumps(updates["history"])
        cols = ", ".join(f"{k}=?" for k in updates)
        vals = list(updates.values()) + [rid]
        with self._conn() as c:
            c.execute(f"UPDATE training_runs SET {cols} WHERE id=?", vals)

    def get_training_history(self, limit: int = 5) -> List[Dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id,started_at,finished_at,status,steps,final_loss,vocab_size FROM training_runs ORDER BY started_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ─── Curiosity Cycles ─────────────────────────────────────────────────────

    def log_curiosity_cycle(self, cycle_num: int, docs: int,
                             found: int, top_score: float):
        with self._conn() as c:
            c.execute(
                "INSERT INTO curiosity_cycles (ts,cycle_num,docs_scanned,insights_found,top_score) VALUES (?,?,?,?,?)",
                (time.time(), cycle_num, docs, found, top_score)
            )
            c.execute("DELETE FROM curiosity_cycles WHERE id NOT IN (SELECT id FROM curiosity_cycles ORDER BY ts DESC LIMIT 500)")

    def get_curiosity_timeline(self, limit: int = 50) -> List[Dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM curiosity_cycles ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ─── Stats Overview ───────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        with self._conn() as c:
            msgs  = c.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            projs = c.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
            docs  = c.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            ins   = c.execute("SELECT COUNT(*) FROM insights").fetchone()[0]
            runs  = c.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0]
            top_s = c.execute("SELECT MAX(curiosity_score) FROM insights").fetchone()[0]
        return {
            "messages":       msgs,
            "projects":       projs,
            "documents":      docs,
            "insights":       ins,
            "training_runs":  runs,
            "top_insight_score": round(top_s or 0, 3),
        }
