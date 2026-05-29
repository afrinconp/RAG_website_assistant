import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from app.config.settings import get_settings


class SQLiteConversationDB:
    """
    Singleton SQLite connection manager.

    Stores conversation history by session ID so the RAG assistant
    can remember previous interactions and use them as context
    in future questions.
    """

    _instance = None

    def __new__(cls) -> "SQLiteConversationDB":
        """
        Create a singleton instance of the database manager.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False

        return cls._instance

    def __init__(self) -> None:
        """
        Initialize the SQLite connection and create tables if needed.
        """
        if self._initialized:
            return

        self.settings = get_settings()

        Path(
            self.settings.sqlite_path
        ).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.conn = sqlite3.connect(
            self.settings.sqlite_path,
            check_same_thread=False,
        )

        self.conn.row_factory = sqlite3.Row

        self._create_tables()
        self._initialized = True

    def _create_tables(self) -> None:
        """
        Create required database tables.
        """
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        self.conn.commit()

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict]] = None,
    ) -> None:
        """
        Store a conversation message.

        Args:
            session_id: Conversation session identifier.
            role: Message role ('user' or 'assistant').
            content: Message content.
            sources: Retrieved source documents.
        """
        self.conn.execute(
            """
            INSERT INTO messages (
                session_id,
                role,
                content,
                sources_json
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                session_id,
                role,
                content,
                json.dumps(
                    sources or [],
                    ensure_ascii=False,
                ),
            ),
        )

        self.conn.commit()

    def get_recent_messages(
        self,
        session_id: str,
        limit: int,
    ) -> List[Dict]:
        """
        Return the most recent messages for a session.

        Args:
            session_id: Conversation identifier.
            limit: Maximum number of messages.

        Returns:
            List of messages ordered chronologically.
        """
        rows = self.conn.execute(
            """
            SELECT
                role,
                content,
                created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (
                session_id,
                limit,
            ),
        ).fetchall()

        return [dict(row) for row in reversed(rows)]

    def get_session_messages(
        self,
        session_id: str,
    ) -> List[Dict]:
        """
        Return all messages for a session.

        Args:
            session_id: Conversation identifier.

        Returns:
            List of messages ordered chronologically.
        """
        rows = self.conn.execute(
            """
            SELECT
                role,
                content,
                sources_json,
                created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()

        return [dict(row) for row in rows]

    def clear_session(
        self,
        session_id: str,
    ) -> None:
        """
        Delete all messages associated with a session.

        Args:
            session_id: Conversation identifier.
        """
        self.conn.execute(
            """
            DELETE FROM messages
            WHERE session_id = ?
            """,
            (session_id,),
        )

        self.conn.commit()

    def all_messages(self) -> List[Dict]:
        """
        Return all stored messages.

        Returns:
            List of all conversation messages.
        """
        rows = self.conn.execute(
            """
            SELECT *
            FROM messages
            ORDER BY created_at ASC
            """
        ).fetchall()

        return [dict(row) for row in rows]