from evallab.chunking import chunk_markdown, load_corpus
from evallab.config import settings


def test_chunks_carry_title_and_section():
    md = "# Doc\n\n## Soft delete\nKept 30 days on Business.\n\n## Other\nText."
    chunks = chunk_markdown(md, "doc.md")
    assert len(chunks) == 2
    assert chunks[0].text.startswith("[Doc > Soft delete]")
    assert chunks[0].metadata["source"] == "doc.md"


def test_long_sections_are_windowed_with_overlap():
    body = " ".join(f"Sentence number {i}." for i in range(200))
    chunks = chunk_markdown(f"# T\n\n## S\n{body}", "t.md", size=300, overlap=50)
    assert len(chunks) > 3
    assert all(len(c.text) <= 300 + 20 for c in chunks)  # + header


def test_chunk_ids_are_stable():
    a = load_corpus(settings.corpus_dir)
    b = load_corpus(settings.corpus_dir)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)


def test_retrieval_finds_the_right_document(offline_pipeline):
    hits, _ = offline_pipeline.retrieve("soft deleted objects restore RestoreObject Business days")
    assert hits[0].source == "03_data_retention_and_deletion.md"


def test_pipeline_returns_contexts_it_used(offline_pipeline):
    r = offline_pipeline.answer("How much does Starter cost per GB-month?")
    assert r.contexts and len(r.contexts) == len(r.sources)
    assert "0.021" in r.answer
    # the generator must have seen exactly these contexts
    sent = offline_pipeline.llm.calls[-1][1]["content"]
    assert all(c in sent for c in r.contexts)
