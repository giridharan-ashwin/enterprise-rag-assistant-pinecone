import sys
from app.services.ingestion import ingest_file

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m app.ingest <file>")
    print(f"Ingested {ingest_file(sys.argv[1])} chunks.")
