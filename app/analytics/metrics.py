from collections import Counter
import re

import pandas as pd

from app.memory.database import SQLiteConversationDB


STOPWORDS = {
    "de", "la", "el", "los", "las", "y", "o", "a", "en", "para", "con", "que",
    "the", "is", "are", "to", "of", "and", "in", "for", "about"
}


def load_messages_df() -> pd.DataFrame:
    db = SQLiteConversationDB()
    rows = db.all_messages()
    if not rows:
        return pd.DataFrame(columns=["id", "session_id", "role", "content", "created_at"])
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["message_length"] = df["content"].str.len()
    df["date"] = df["created_at"].dt.date
    return df


def top_user_terms(df: pd.DataFrame, n: int = 15):
    if df.empty:
        return []
    text = " ".join(df[df["role"] == "user"]["content"].fillna(""))
    words = re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñ]{4,}", text.lower())
    words = [w for w in words if w not in STOPWORDS]
    return Counter(words).most_common(n)
