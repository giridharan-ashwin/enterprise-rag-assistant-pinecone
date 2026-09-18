from pathlib import Path
from app.services.ingestion import ingest_file
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"stress_test"
files=sorted(DATA.glob("*.md"))
print(f"Found {len(files)} documents")
total=0
for p in files:
    n=ingest_file(str(p)); total+=n; print(f"{p.name}: {n} chunks")
print(f"Total chunks ingested: {total}")
