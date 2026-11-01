
import sqlite3
from datetime import datetime

DB_NAME = "scamshield.db"


def create_table():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_type TEXT,
            input_text TEXT,
            result TEXT,
            confidence REAL,
            timestamp TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_detection(detection_type, input_text, result, confidence):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO history
        (detection_type, input_text, result, confidence, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (
        detection_type,
        input_text,
        result,
        confidence,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


def get_history():
    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()
    cursor.execute("SELECT * FROM history ORDER BY id DESC")

    rows = cursor.fetchall()
    conn.close()

    return rows

def delete_history():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM history")

    conn.commit()
    conn.close()