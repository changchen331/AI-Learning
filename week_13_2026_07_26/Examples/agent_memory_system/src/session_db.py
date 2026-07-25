"""
Layer 2 短期记忆：SQLite 会话历史管理

教学重点：
  1. 会话（Session）与消息（Message）的两表结构
  2. 当日会话历史如何注入 Context Window
  3. 跨会话的历史检索（非语义，按时间顺序）

使用方式：
  from src.session_db import SessionDB
  db = SessionDB()
  sid = db.new_session()
  db.add_message(sid, "user", "你好")
  messages = db.get_session_messages(sid)
"""

import sqlite3
from datetime import datetime, date
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "outputs" / "sessions" / "memory.db"


class SessionDB:
    def __init__(self, db_path: Path = DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
                               CREATE TABLE IF NOT EXISTS sessions
                               (
                                   id
                                   INTEGER
                                   PRIMARY
                                   KEY
                                   AUTOINCREMENT,
                                   start_time
                                   TEXT
                                   NOT
                                   NULL,
                                   end_time
                                   TEXT,
                                   title
                                   TEXT,
                                   flushed
                                   INTEGER
                                   DEFAULT
                                   0
                               );
                               CREATE TABLE IF NOT EXISTS messages
                               (
                                   id
                                   INTEGER
                                   PRIMARY
                                   KEY
                                   AUTOINCREMENT,
                                   session_id
                                   INTEGER
                                   NOT
                                   NULL,
                                   role
                                   TEXT
                                   NOT
                                   NULL,
                                   content
                                   TEXT
                                   NOT
                                   NULL,
                                   timestamp
                                   TEXT
                                   NOT
                                   NULL,
                                   FOREIGN
                                   KEY
                               (
                                   session_id
                               ) REFERENCES sessions
                               (
                                   id
                               )
                                   );
                               """)

    # ── 会话管理 ─────────────────────────────────────────────────────────────

    def new_session(self) -> int:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO sessions (start_time) VALUES (?)", (now,)
            )
            return cur.lastrowid

    def close_session(self, session_id: int, title: str = None):
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET end_time=?, title=? WHERE id=?",
                (now, title, session_id),
            )

    def mark_flushed(self, session_id: int):
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET flushed=1 WHERE id=?", (session_id,)
            )

    # ── 消息管理 ─────────────────────────────────────────────────────────────

    def add_message(self, session_id: int, role: str, content: str):
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
                (session_id, role, content, now),
            )

    def get_session_messages(self, session_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, timestamp FROM messages WHERE session_id=? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_message_count(self, session_id: int) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM messages WHERE session_id=?",
                (session_id,),
            ).fetchone()
        return row["cnt"]

    # ── 历史检索（跨会话）────────────────────────────────────────────────────

    def get_recent_sessions(self, limit: int = 5) -> list[dict]:
        """返回最近 N 个已关闭的会话摘要（含消息数）"""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.id,
                       s.start_time,
                       s.title,
                       s.flushed,
                       COUNT(m.id) as msg_count
                FROM sessions s
                         LEFT JOIN messages m ON m.session_id = s.id
                WHERE s.end_time IS NOT NULL
                GROUP BY s.id
                ORDER BY s.id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_today_messages(self) -> list[dict]:
        """返回今日全部消息（Layer 2 注入用）"""
        today = date.today().isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT m.role, m.content, m.timestamp, m.session_id
                FROM messages m
                         JOIN sessions s ON s.id = m.session_id
                WHERE DATE (m.timestamp) = ?
                ORDER BY m.id
                """,
                (today,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_unflushed_sessions(self) -> list[int]:
        """返回尚未 flush 的已关闭会话 ID"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM sessions WHERE end_time IS NOT NULL AND flushed=0 ORDER BY id",
            ).fetchall()
        return [r["id"] for r in rows]
