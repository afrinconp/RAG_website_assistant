from app.memory.database import SQLiteConversationDB


def test_database_has_session_memory_methods():
    db = SQLiteConversationDB()
    assert hasattr(db, "get_session_messages")
    assert hasattr(db, "clear_session")
