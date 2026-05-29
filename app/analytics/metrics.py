from collections import Counter
import re
from typing import List, Tuple

import pandas as pd

from app.memory.database import SQLiteConversationDB


STOPWORDS = {
    "de",
    "la",
    "el",
    "los",
    "las",
    "y",
    "o",
    "a",
    "en",
    "para",
    "con",
    "que",
    "the",
    "is",
    "are",
    "to",
    "of",
    "and",
    "in",
    "for",
    "about",
}

EMPTY_COLUMNS = [
    "id",
    "session_id",
    "role",
    "content",
    "created_at",
]


def load_messages_df() -> pd.DataFrame:
    """
    Load all conversation messages from SQLite and return a DataFrame.

    Returns:
        pd.DataFrame: Messages enriched with date and length metrics.
    """
    db = SQLiteConversationDB()
    rows = db.all_messages()

    if not rows:
        return pd.DataFrame(columns=EMPTY_COLUMNS)

    df = pd.DataFrame(rows)

    df["created_at"] = pd.to_datetime(df["created_at"])
    df["message_length"] = df["content"].str.len()
    df["date"] = df["created_at"].dt.date

    return df


def top_user_terms(
    df: pd.DataFrame,
    n: int = 15,
) -> List[Tuple[str, int]]:
    """
    Return the most frequent user terms excluding stopwords.

    Args:
        df: Conversation DataFrame.
        n: Number of terms to return.

    Returns:
        List[Tuple[str, int]]: Most common terms and their frequencies.
    """
    if df.empty:
        return []

    text = " ".join(
        df[df["role"] == "user"]["content"].fillna("")
    )

    words = re.findall(
        r"[A-Za-zÁÉÍÓÚáéíóúÑñ]{4,}",
        text.lower(),
    )

    filtered_words = [
        word
        for word in words
        if word not in STOPWORDS
    ]

    return Counter(filtered_words).most_common(n)
