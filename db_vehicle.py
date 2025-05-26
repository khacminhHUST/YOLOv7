import sqlite3

def init_db():
    conn = sqlite3.connect("vehicle.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS vehicle (
            id TEXT PRIMARY KEY,
            plate_text TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_vehicle(id, plate_text):
    conn = sqlite3.connect("vehicle.db")
    cur = conn.cursor()
    cur.execute("INSERT INTO vehicle (id, plate_text) VALUES (?, ?)", (id, plate_text))
    conn.commit()
    conn.close()

def get_plate_by_id(id):
    conn = sqlite3.connect("vehicle.db")
    cur = conn.cursor()
    cur.execute("SELECT plate_text FROM vehicle WHERE id = ?", (id,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None
