from app.memory.database import SQLiteConversationDB


def test_database_has_session_memory_methods() -> None:
    """
    Verify that the database exposes the methods required
    for session memory management.
    """
    # Arrange
    database = SQLiteConversationDB()

    # Assert
    assert hasattr(
        database,
        "get_session_messages",
    )

    assert hasattr(
        database,
        "clear_session",
    )