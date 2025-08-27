import sqlite3

def init_db():
    conn = sqlite3.connect("meetings.db")
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        email TEXT PRIMARY KEY,
        created_at TEXT
    )
    """)
    
    # Create meetings table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS meetings (
        id TEXT PRIMARY KEY,
        title TEXT,
        user_email TEXT,
        original_file TEXT,
        audio_file TEXT,
        transcription TEXT,
        summary TEXT,
        key_topics TEXT,
        status TEXT,
        created_at TEXT,
        completed_at TEXT,
        FOREIGN KEY (user_email) REFERENCES users (email)
    )
    """)
    
    # Create action_items table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS action_items (
        id TEXT PRIMARY KEY,
        meeting_id TEXT,
        description TEXT,
        assignee TEXT,
        due_date TEXT,
        status TEXT,
        created_at TEXT,
        FOREIGN KEY (meeting_id) REFERENCES meetings (id)
    )
    """)
    
    # Create decisions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS decisions (
        id TEXT PRIMARY KEY,
        meeting_id TEXT,
        description TEXT,
        created_at TEXT,
        FOREIGN KEY (meeting_id) REFERENCES meetings (id)
    )
    """)
    
    conn.commit()
    conn.close()