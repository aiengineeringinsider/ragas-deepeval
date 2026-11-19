from fastapi.testclient import TestClient

from evallab import store
from evallab.records import EvalRecord


def _rec(i, s):
    return EvalRecord(id=i, question="q?", reference="r", category="single_hop", answer="a",
                      contexts=["c"], sources=["s.md"], reference_sources=["s.md"],
                      latency_s=1.0, model="fake", scores={"faithfulness": s})


def test_store_roundtrip(tmp_db):
    run_id = store.save_run([_rec("g001", 0.9), _rec("g002", float("nan"))], "t", "gen", "judge",
                            {"scores": {}}, db_path=tmp_db)
    rows = store.run_results(run_id, db_path=tmp_db)
    assert {r["item_id"] for r in rows} == {"g001", "g002"}
    assert [r["score"] for r in rows if r["item_id"] == "g002"] == [None]  # NaN stored as NULL
    store.add_human_label(run_id, "g001", "ann", 1, 5, db_path=tmp_db)
    assert len(store.human_labels(run_id, db_path=tmp_db)) == 1


def test_api_ask_uses_injected_pipeline(offline_pipeline, monkeypatch, tmp_path):
    from api import main

    main.app.dependency_overrides[main.get_pipeline] = lambda: offline_pipeline
    client = TestClient(main.app)
    r = client.post("/ask", json={"question": "How much does Starter cost per GB-month?"})
    assert r.status_code == 200
    body = r.json()
    assert "0.021" in body["answer"]
    assert body["checks"]["numeric_grounding"] == 1.0
    assert client.post("/ask", json={"question": ""}).status_code == 422
    main.app.dependency_overrides.clear()


def test_api_rejects_invalid_human_label():
    from api import main

    client = TestClient(main.app)
    bad = {"run_id": "r", "item_id": "g001", "annotator": "a", "faithful": 3, "score": 9}
    assert client.post("/human-labels", json=bad).status_code == 422
