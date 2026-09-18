from pathlib import Path

from app.services.chunker import chunk_text


text = Path("data/sample/company_handbook.md").read_text()

chunks = chunk_text(
    text=text,
    chunk_size=900,
    overlap=120,
)

print("Chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- CHUNK {i} ---")
    print("Section:", chunk["section"])
    print("Text:", chunk["text"])
