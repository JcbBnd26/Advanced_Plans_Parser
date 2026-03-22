"""Temporary script to inspect corrections.db."""

import sqlite3

db = sqlite3.connect(r"c:\Projects\Advanced_Plan_Parser\data\corrections.db")
cur = db.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
print("Tables:", tables)

for t in tables:
    cur.execute(f"SELECT COUNT(*) FROM [{t}]")
    print(f"  {t}: {cur.fetchone()[0]} rows")

# Check detections specifically
if "detections" in tables:
    cur.execute("SELECT DISTINCT run_id FROM detections ORDER BY run_id DESC LIMIT 5")
    runs = cur.fetchall()
    print("\nLatest run_ids in detections:", [r[0] for r in runs])

    cur.execute("SELECT DISTINCT doc_id, page FROM detections ORDER BY doc_id, page")
    pages = cur.fetchall()
    print("Pages with detections:", pages[:20])

    cur.execute("SELECT element_type, COUNT(*) FROM detections GROUP BY element_type")
    types = cur.fetchall()
    print("Detection types:", dict(types))

# Check documents table
if "documents" in tables:
    cur.execute("SELECT * FROM documents")
    docs = cur.fetchall()
    print("\nDocuments:", docs[:10])

db.close()
