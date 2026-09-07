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
from typing import List, Dict, Optional, Tuple, Any
from contextlib import contextmanager
import numpy as np

from jarvis_localhost.logging_config import get_logger
from jarvis_localhost.paths import DATA_ROOT, REPO_ROOT

logger = get_logger(__name__)

DB_PATH = DATA_ROOT / "jarvis.db"
LEGACY_DB_PATH = REPO_ROOT / "data" / "jarvis.db"


class JarvisDB:
    """Thread-safe SQLite wrapper for J.A.R.V.I.S persistence."""

    def __init__(self, path: str | Path = DB_PATH):
        destination = Path(path).resolve(strict=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if (
            destination == DB_PATH.resolve(strict=False)
            and not destination.exists()
            and LEGACY_DB_PATH.is_file()
        ):
            self._migrate_legacy_database(LEGACY_DB_PATH, destination)
        self.path = str(destination)
        self._init_schema()
        logger.info("[DB] SQLite initialized -> %s", destination)

    @staticmethod
    def _migrate_legacy_database(source: Path, destination: Path) -> None:
        """Copy a live legacy SQLite database using SQLite's backup API."""

        temporary = destination.with_suffix(destination.suffix + ".migration.tmp")
        temporary.unlink(missing_ok=True)
        source_connection = sqlite3.connect(str(source), timeout=30.0)
        destination_connection = sqlite3.connect(str(temporary), timeout=30.0)
        try:
            source_connection.backup(destination_connection)
            check = destination_connection.execute("PRAGMA quick_check").fetchone()
            if not check or check[0] != "ok":
                raise sqlite3.DatabaseError("legacy database backup failed validation")
            destination_connection.close()
            source_connection.close()
            temporary.replace(destination)
            return
        finally:
            try:
                destination_connection.close()
            finally:
                source_connection.close()
            temporary.unlink(missing_ok=True)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=268435456")
        conn.execute("PRAGMA cache_size=-64000")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
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
                    document_sha256 TEXT,
                    corpus_sha256 TEXT,
                    chunks_file TEXT,
                    canonical_chunks INTEGER DEFAULT 0,
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
                    history     TEXT,                  -- JSON array of step logs
                    corpus_sha256 TEXT,
                    tokenizer_sha256 TEXT,
                    retriever_loss REAL,
                    backend TEXT,
                    profile TEXT,
                    error TEXT
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

                -- ─── Linguistic & Cognitive Memory ────────────────────────────────────
                CREATE TABLE IF NOT EXISTS linguistic_profiles (
                    chunk_id               TEXT PRIMARY KEY,
                    document_id            TEXT NOT NULL,
                    ttr                    REAL,
                    yule_k                 REAL,
                    mean_sentence_length   REAL,
                    sentence_count         INTEGER,
                    colon_density          REAL,
                    semicolon_density      REAL,
                    math_density           REAL,
                    list_density           REAL,
                    connector_density      REAL,
                    primary_discourse      TEXT,
                    rhetorical_role        TEXT,
                    rhetorical_confidence  REAL,
                    style_vector           BLOB,
                    rhetorical_vector      BLOB,
                    profile_json           TEXT,
                    created_at             REAL NOT NULL
                );

                -- ─── Interaction Feedback & Preferences (Behavior Memory) ───────────
                CREATE TABLE IF NOT EXISTS interaction_feedback (
                    interaction_id    TEXT PRIMARY KEY,
                    session_id        TEXT NOT NULL,
                    question          TEXT NOT NULL,
                    answer            TEXT NOT NULL,
                    sources           TEXT,
                    accepted          INTEGER DEFAULT 1,
                    explicit_rating   INTEGER,
                    regenerated       INTEGER DEFAULT 0,
                    implicit_reward   REAL DEFAULT 0.0,
                    intent            TEXT,
                    outline           TEXT,
                    created_at        REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS preference_pairs (
                    id                TEXT PRIMARY KEY,
                    prompt            TEXT NOT NULL,
                    chosen_answer     TEXT NOT NULL,
                    rejected_answer   TEXT NOT NULL,
                    chosen_sources    TEXT,
                    rejected_sources  TEXT,
                    reward_delta      REAL DEFAULT 1.0,
                    created_at        REAL NOT NULL
                );

                -- Indices
                CREATE INDEX IF NOT EXISTS idx_messages_ts      ON messages(ts DESC);
                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
                CREATE INDEX IF NOT EXISTS idx_insights_score   ON insights(curiosity_score DESC);
                CREATE INDEX IF NOT EXISTS idx_metrics_ts       ON metrics(ts DESC);
                CREATE INDEX IF NOT EXISTS idx_documents_ts     ON documents(ts DESC);
                CREATE INDEX IF NOT EXISTS idx_ling_role        ON linguistic_profiles(rhetorical_role);
                CREATE INDEX IF NOT EXISTS idx_ling_doc         ON linguistic_profiles(document_id);
                CREATE INDEX IF NOT EXISTS idx_feedback_ts      ON interaction_feedback(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_pref_ts          ON preference_pairs(created_at DESC);
            """)
            self._migrate_columns(c)
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_sha "
                "ON documents(document_sha256)"
            )

    @staticmethod
    def _migrate_columns(connection: sqlite3.Connection) -> None:
        """Add sovereign-lineage fields without invalidating existing rows."""

        migrations = {
            "documents": {
                "document_sha256": "TEXT",
                "corpus_sha256": "TEXT",
                "chunks_file": "TEXT",
                "canonical_chunks": "INTEGER DEFAULT 0",
            },
            "training_runs": {
                "corpus_sha256": "TEXT",
                "tokenizer_sha256": "TEXT",
                "retriever_loss": "REAL",
                "backend": "TEXT",
                "profile": "TEXT",
                "error": "TEXT",
            },
            "insights": {
                "page": "INTEGER DEFAULT 0",
                "chunk_id": "TEXT",
                "document_sha256": "TEXT",
                "intrinsic_reward": "REAL DEFAULT 0",
            },
        }
        for table, columns in migrations.items():
            existing = {
                str(row[1])
                for row in connection.execute(f"PRAGMA table_info({table})")
            }
            for column, declaration in columns.items():
                if column not in existing:
                    connection.execute(
                        f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
                    )

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
                """INSERT INTO insights (
                       id, source, chunk_text, summary, tags,
                       curiosity_score, novelty_score, entropy_score,
                       surprise_score, connections, times_surfaced, is_new, ts,
                       page, chunk_id, document_sha256, intrinsic_reward
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                       source=excluded.source,
                       chunk_text=excluded.chunk_text,
                       summary=excluded.summary,
                       tags=excluded.tags,
                       curiosity_score=excluded.curiosity_score,
                       novelty_score=excluded.novelty_score,
                       entropy_score=excluded.entropy_score,
                       surprise_score=excluded.surprise_score,
                       connections=excluded.connections,
                       page=excluded.page,
                       chunk_id=excluded.chunk_id,
                       document_sha256=excluded.document_sha256,
                       intrinsic_reward=excluded.intrinsic_reward,
                       ts=excluded.ts""",
                (ins["id"], ins["source"], ins["chunk_text"][:800],
                 ins["summary"], json.dumps(ins.get("tags", [])),
                 ins.get("curiosity_score", 0), ins.get("novelty_score", 0),
                 ins.get("entropy_score", 0),   ins.get("surprise_score", 0),
                 json.dumps(ins.get("connections", [])),
                 ins.get("times_surfaced", 0),  int(ins.get("is_new", True)),
                 ins.get("timestamp", time.time()), ins.get("page", 0),
                 ins.get("chunk_id", ""), ins.get("document_sha256", ""),
                 ins.get("intrinsic_reward", 0.0))
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

    def save_document(
        self,
        filename: str,
        path: str,
        stats: Dict,
        corpus: str = "",
        *,
        document_id: str = "",
        document_sha256: str = "",
        corpus_sha256: str = "",
        chunks_file: str = "",
        canonical_chunks: int = 0,
    ) -> str:
        did = document_id or str(uuid.uuid4())[:8]
        with self._conn() as c:
            c.execute(
                """INSERT INTO documents (
                       id, filename, path, pages, words, tables, images,
                       language, indexed, corpus, document_sha256,
                       corpus_sha256, chunks_file, canonical_chunks, ts
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                       filename=excluded.filename,
                       path=excluded.path,
                       pages=excluded.pages,
                       words=excluded.words,
                       tables=excluded.tables,
                       images=excluded.images,
                       language=excluded.language,
                       indexed=0,
                       corpus=excluded.corpus,
                       document_sha256=excluded.document_sha256,
                       corpus_sha256=excluded.corpus_sha256,
                       chunks_file=excluded.chunks_file,
                       canonical_chunks=excluded.canonical_chunks,
                       ts=excluded.ts""",
                (did, filename, path, stats.get("pages", 0),
                 stats.get("words", 0), stats.get("tables", 0),
                 stats.get("images", 0), stats.get("language", ""),
                 0, corpus, document_sha256, corpus_sha256, chunks_file,
                 int(canonical_chunks), time.time())
            )
        return did

    def list_documents(self) -> List[Dict]:
        with self._conn() as c:
            rows = c.execute(
                """SELECT id,filename,path,pages,words,tables,images,language,
                          indexed,document_sha256,corpus_sha256,chunks_file,
                          canonical_chunks,ts
                   FROM documents ORDER BY ts DESC"""
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_indexed(self, doc_id: str):
        with self._conn() as c:
            c.execute("UPDATE documents SET indexed=1 WHERE id=?", (doc_id,))

    # ─── Training Runs ────────────────────────────────────────────────────────

    def start_training_run(
        self,
        *,
        corpus_sha256: str = "",
        backend: str = "",
        profile: Optional[Dict] = None,
    ) -> str:
        rid = str(uuid.uuid4())[:8]
        with self._conn() as c:
            c.execute(
                """INSERT INTO training_runs (
                       id,started_at,status,history,corpus_sha256,backend,profile
                   ) VALUES (?,?,?,?,?,?,?)""",
                (
                    rid,
                    time.time(),
                    "running",
                    "[]",
                    corpus_sha256,
                    backend,
                    json.dumps(profile or {}, ensure_ascii=False),
                )
            )
        return rid

    def update_training_run(self, rid: str, **kwargs):
        allowed = {
            "finished_at",
            "status",
            "steps",
            "final_loss",
            "vocab_size",
            "history",
            "corpus_sha256",
            "tokenizer_sha256",
            "retriever_loss",
            "backend",
            "profile",
            "error",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        for json_field in ("history", "profile"):
            if json_field in updates and isinstance(
                updates[json_field], (list, dict)
            ):
                updates[json_field] = json.dumps(
                    updates[json_field], ensure_ascii=False
                )
        cols = ", ".join(f"{k}=?" for k in updates)
        vals = list(updates.values()) + [rid]
        with self._conn() as c:
            c.execute(f"UPDATE training_runs SET {cols} WHERE id=?", vals)

    def get_training_history(self, limit: int = 5) -> List[Dict]:
        with self._conn() as c:
            rows = c.execute(
                """SELECT id,started_at,finished_at,status,steps,final_loss,
                          vocab_size,corpus_sha256,tokenizer_sha256,
                          retriever_loss,backend,profile,error
                   FROM training_runs ORDER BY started_at DESC LIMIT ?""",
                (limit,)
            ).fetchall()
        result = []
        for row in rows:
            payload = dict(row)
            try:
                payload["profile"] = json.loads(payload.get("profile") or "{}")
            except (TypeError, json.JSONDecodeError):
                payload["profile"] = {}
            result.append(payload)
        return result

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
            lings = c.execute("SELECT COUNT(*) FROM linguistic_profiles").fetchone()[0]
            feedbacks = c.execute("SELECT COUNT(*) FROM interaction_feedback").fetchone()[0]
            prefs = c.execute("SELECT COUNT(*) FROM preference_pairs").fetchone()[0]
        return {
            "messages":       msgs,
            "projects":       projs,
            "documents":      docs,
            "insights":       ins,
            "training_runs":  runs,
            "top_insight_score": round(top_s or 0, 3),
            "linguistic_profiles": lings,
            "interaction_feedbacks": feedbacks,
            "preference_pairs": prefs,
        }

    # ─── Linguistic Profiles ───────────────────────────────────────────────────

    def save_linguistic_profiles_batch(
        self, records: List[Tuple[str, str, Any]]
    ) -> None:
        """Save a batch of (chunk_id, document_id, LinguisticProfile)."""
        now = time.time()
        with self._conn() as c:
            for chunk_id, document_id, profile in records:
                style_blob = sqlite3.Binary(
                    np.array(profile.style_vector, dtype=np.float32).tobytes()
                )
                rhet_blob = sqlite3.Binary(
                    np.array(profile.rhetorical_vector, dtype=np.float32).tobytes()
                )
                profile_json = json.dumps(profile.to_dict(), ensure_ascii=False)
                c.execute(
                    """
                    INSERT OR REPLACE INTO linguistic_profiles (
                        chunk_id, document_id, ttr, yule_k,
                        mean_sentence_length, sentence_count, colon_density,
                        semicolon_density, math_density, list_density,
                        connector_density, primary_discourse, rhetorical_role,
                        rhetorical_confidence, style_vector, rhetorical_vector,
                        profile_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        document_id,
                        profile.lexical.ttr,
                        profile.lexical.yule_k,
                        profile.syntactic.mean_sentence_length,
                        profile.syntactic.sentence_count,
                        profile.syntactic.colon_density,
                        profile.syntactic.semicolon_density,
                        profile.syntactic.math_density,
                        profile.syntactic.list_density,
                        profile.discourse.connective_density,
                        profile.discourse.primary_discourse_function,
                        profile.rhetorical.primary_role.value,
                        profile.rhetorical.confidence,
                        style_blob,
                        rhet_blob,
                        profile_json,
                        now,
                    ),
                )

    def get_linguistic_profile(self, chunk_id: str) -> Optional[Dict]:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM linguistic_profiles WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
        if not row:
            return None
        payload = dict(row)
        if payload.get("profile_json"):
            try:
                payload["details"] = json.loads(payload["profile_json"])
            except Exception:
                payload["details"] = {}
        payload.pop("style_vector", None)
        payload.pop("rhetorical_vector", None)
        return payload

    def get_linguistics_summary(self) -> Dict[str, Any]:
        with self._conn() as c:
            total = c.execute("SELECT COUNT(*) FROM linguistic_profiles").fetchone()[0]
            if total == 0:
                return {"total_analyzed_chunks": 0, "rhetorical_distribution": {}}
            roles = c.execute(
                "SELECT rhetorical_role, COUNT(*) FROM linguistic_profiles GROUP BY rhetorical_role"
            ).fetchall()
            avg_ttr = c.execute("SELECT AVG(ttr) FROM linguistic_profiles").fetchone()[0] or 0.0
            avg_len = c.execute("SELECT AVG(mean_sentence_length) FROM linguistic_profiles").fetchone()[0] or 0.0
            avg_math = c.execute("SELECT AVG(math_density) FROM linguistic_profiles").fetchone()[0] or 0.0
            avg_connectors = c.execute("SELECT AVG(connector_density) FROM linguistic_profiles").fetchone()[0] or 0.0
        return {
            "total_analyzed_chunks": total,
            "average_ttr": round(avg_ttr, 4),
            "average_sentence_length": round(avg_len, 2),
            "average_math_density": round(avg_math, 4),
            "average_connector_density": round(avg_connectors, 4),
            "rhetorical_distribution": {row[0]: row[1] for row in roles},
        }

    # ─── Behavior & Preference Memory ─────────────────────────────────────────

    def save_interaction_feedback(
        self,
        interaction_id: str,
        session_id: str,
        question: str,
        answer: str,
        sources: Optional[List[Dict]] = None,
        accepted: bool = True,
        explicit_rating: Optional[int] = None,
        regenerated: bool = False,
        implicit_reward: float = 0.0,
        intent: Optional[str] = None,
        outline: Optional[List[str]] = None,
    ) -> None:
        with self._conn() as c:
            c.execute(
                """
                INSERT OR REPLACE INTO interaction_feedback (
                    interaction_id, session_id, question, answer, sources,
                    accepted, explicit_rating, regenerated, implicit_reward,
                    intent, outline, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    interaction_id,
                    session_id,
                    question,
                    answer,
                    json.dumps(sources or [], ensure_ascii=False),
                    1 if accepted else 0,
                    explicit_rating,
                    1 if regenerated else 0,
                    implicit_reward,
                    intent,
                    json.dumps(outline or [], ensure_ascii=False),
                    time.time(),
                ),
            )

    def save_preference_pair(
        self,
        prompt: str,
        chosen_answer: str,
        rejected_answer: str,
        chosen_sources: Optional[List[Dict]] = None,
        rejected_sources: Optional[List[Dict]] = None,
        reward_delta: float = 1.0,
    ) -> str:
        pair_id = f"pref_{uuid.uuid4().hex[:16]}"
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO preference_pairs (
                    id, prompt, chosen_answer, rejected_answer,
                    chosen_sources, rejected_sources, reward_delta, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pair_id,
                    prompt,
                    chosen_answer,
                    rejected_answer,
                    json.dumps(chosen_sources or [], ensure_ascii=False),
                    json.dumps(rejected_sources or [], ensure_ascii=False),
                    reward_delta,
                    time.time(),
                ),
            )
        return pair_id

    def get_preference_pairs(self, limit: int = 100) -> List[Dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM preference_pairs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        result = []
        for row in rows:
            p = dict(row)
            try:
                p["chosen_sources"] = json.loads(p.get("chosen_sources") or "[]")
                p["rejected_sources"] = json.loads(p.get("rejected_sources") or "[]")
            except Exception:
                pass
            result.append(p)
        return result
