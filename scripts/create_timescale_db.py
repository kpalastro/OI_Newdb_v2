#!/usr/bin/env python3
"""
Create TimescaleDB database oi_db using credentials from .env.
- Creates database oi_db if it does not exist.
- Connects to oi_db and enables the TimescaleDB extension.
"""

import os
import sys

# Load .env from project root (parent of scripts/)
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_env_path = os.path.join(_root, ".env")
if not os.path.isfile(_env_path):
    print(f"Error: .env not found at {_env_path}")
    sys.exit(1)

from dotenv import load_dotenv
load_dotenv(_env_path)

host = os.getenv("OI_TRACKER_DB_HOST", "localhost")
port = int(os.getenv("OI_TRACKER_DB_PORT", "5432"))
name = os.getenv("OI_TRACKER_DB_NAME", "oi_db")
user = os.getenv("OI_TRACKER_DB_USER", "")
password = os.getenv("OI_TRACKER_DB_PASSWORD", "")

if not user or not password:
    print("Error: OI_TRACKER_DB_USER and OI_TRACKER_DB_PASSWORD must be set in .env")
    sys.exit(1)

try:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
except ImportError:
    print("Error: psycopg2 is required. Install with: pip install psycopg2-binary")
    sys.exit(1)


def main():
    # 1. Connect to default 'postgres' DB to create oi_db
    print(f"Connecting to PostgreSQL at {host}:{port} as {user}...")
    conn = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database="postgres",
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,))
    if cur.fetchone():
        print(f"Database '{name}' already exists.")
    else:
        # Use template0 to avoid collation version issues (e.g. after PG/glibc upgrade)
        cur.execute(f'CREATE DATABASE "{name}" WITH TEMPLATE template0')
        print(f"Created database '{name}'.")

    cur.close()
    conn.close()

    # 2. Connect to oi_db and enable TimescaleDB
    print(f"Connecting to '{name}' to enable TimescaleDB extension...")
    conn = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=name,
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
        print("TimescaleDB extension enabled (or already present).")
    except Exception as e:
        print(f"Warning: Could not enable TimescaleDB extension: {e}")
        print("Ensure TimescaleDB is installed on the server. Database oi_db is ready for plain PostgreSQL.")

    cur.close()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
