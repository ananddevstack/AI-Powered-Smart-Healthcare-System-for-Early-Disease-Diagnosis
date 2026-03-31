import sqlite3

# Create database & table
def init_chat_db():
    conn = sqlite3.connect("chat.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender TEXT,
            receiver TEXT,
            message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# Send message
def send_message(sender, receiver, message):
    conn = sqlite3.connect("chat.db")
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages (sender, receiver, message) VALUES (?, ?, ?)",
        (sender, receiver, message)
    )
    conn.commit()
    conn.close()

# Fetch messages
def get_messages(user1, user2):
    conn = sqlite3.connect("chat.db")
    cur = conn.cursor()
    cur.execute("""
        SELECT sender, message, timestamp FROM messages
        WHERE (sender=? AND receiver=?) OR (sender=? AND receiver=?)
        ORDER BY timestamp
    """, (user1, user2, user2, user1))
    data = cur.fetchall()
    conn.close()
    return data
