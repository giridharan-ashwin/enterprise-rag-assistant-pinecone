from app.services.chunker import chunk_text

def test_chunk_text_produces_multiple_chunks():
    chunks = chunk_text("A" * 250, 100, 20)
    assert len(chunks) >= 3

def test_empty_text_returns_empty_list():
    assert chunk_text("   ", 100, 20) == []

