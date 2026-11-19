"""Streamlit evaluation dashboard (Modules 13 and 14).

Run:  streamlit run dashboard/app.py
Reads the same SQLite store the CLI and API write to, so it works even when
Ollama is not running (only the "Ask" page needs a live model).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from evallab import store  # noqa: E402
from evallab.config import settings  # noqa: E402
from evallab.stats import binary_agreement, bootstrap_ci, kappa_label  # noqa: E402

st.set_page_config(page_title="Local LLM Eval Platform", layout="wide")


@st.cache_data(ttl=30)
def load_runs() -> pd.DataFrame:
    df = pd.DataFrame(store.list_runs())
    if df.empty:
        return df
    df["scores"] = df["summary_json"].map(lambda s: json.loads(s).get("scores", {}))
    return df


@st.cache_data(ttl=30)
def load_results(run_id: str) -> pd.DataFrame:
    return pd.DataFrame(store.run_results(run_id))


def metric_table(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, g in results.groupby("metric"):
        vals = [v for v in g["score"].tolist() if v is not None and not math.isnan(v)]
        mean, lo, hi = bootstrap_ci(vals) if vals else (float("nan"),) * 3
        rows.append({"metric": metric, "mean": mean, "ci95_low": lo, "ci95_high": hi,
                     "n_scored": len(vals), "n_missing": len(g) - len(vals)})
    return pd.DataFrame(rows).sort_values("metric")


page = st.sidebar.radio("Page", ["Overview", "Run detail", "Judge comparison",
                                 "Human labeling", "Ask"])
runs = load_runs()

# ------------------------------------------------------------------ Overview
if page == "Overview":
    st.title("Evaluation runs")
    if runs.empty:
        st.info("No runs yet. Run `python scripts/11_full_run.py` or POST /evaluate.")
        st.stop()
    wide = pd.json_normalize(runs["scores"]).set_index(runs["run_id"])
    show = runs[["run_id", "created_at", "label", "generator", "judge"]].set_index("run_id").join(wide)
    st.dataframe(show, width="stretch")
    metrics = [c for c in wide.columns if c != "p50_latency_s"]
    pick = st.multiselect("Trend", metrics, default=[m for m in ["faithfulness", "context_recall",
                                                                 "refusal_correctness"] if m in metrics])
    if pick:
        trend = show.sort_values("created_at").set_index("created_at")[pick]
        if len(trend) < 2:
            # A line needs two runs; show the single run as bars instead of an empty chart.
            single = pd.DataFrame({"score": trend.iloc[0].astype(float)})
            st.bar_chart(single, horizontal=True)
            st.caption("Only one run so far. Run the full evaluation again to see a trend.")
        else:
            st.line_chart(trend)
    st.caption("Scores are judge estimates with sampling noise; compare runs with the paired "
               "regression gate, not by eyeballing a single mean.")

# ---------------------------------------------------------------- Run detail
elif page == "Run detail":
    if runs.empty:
        st.info("No runs yet.")
        st.stop()
    run_id = st.selectbox("Run", runs["run_id"])
    res = load_results(run_id)
    st.subheader("Metric summary (mean with 95% bootstrap CI)")
    st.dataframe(metric_table(res).round(3), width="stretch")

    cats = sorted(res["category"].unique())
    cat = st.multiselect("Category", cats, default=cats)
    metric = st.selectbox("Sort by metric", sorted(res["metric"].unique()))
    sub = res[(res["metric"] == metric) & (res["category"].isin(cat))].sort_values("score")
    st.subheader(f"Items by {metric} (worst first)")
    st.dataframe(sub[["item_id", "category", "score", "question", "answer", "reason"]],
                 width="stretch")
    item = st.selectbox("Inspect item", sub["item_id"])
    row = res[res["item_id"] == item].iloc[0]
    st.markdown(f"**Question.** {row['question']}")
    st.markdown(f"**Answer.** {row['answer']}")
    st.markdown(f"**Reference.** {row['reference']}")
    with st.expander("Retrieved contexts"):
        for i, (c, s) in enumerate(zip(json.loads(row["contexts_json"]),
                                       json.loads(row["sources_json"])), start=1):
            st.markdown(f"**[{i}] {s}**")
            st.text(c)
    st.dataframe(res[res["item_id"] == item][["metric", "score", "reason"]], width="stretch")

# ---------------------------------------------------------- Judge comparison
elif page == "Judge comparison":
    st.title("Local judges vs human labels")
    p = ROOT / "results" / "judge_comparison.json"
    if not p.exists():
        st.info("Run `python scripts/07_compare_judges.py` first.")
        st.stop()
    df = pd.DataFrame(json.loads(p.read_text()))
    cols = ["model", "accuracy", "f1_faithful", "hallucination_recall", "cohen_kappa",
            "spearman_rho", "self_consistency", "latency_p50", "parse_failures"]
    st.dataframe(df[[c for c in cols if c in df.columns]].round(3), width="stretch")
    # stack=False: these are three separate rates on a 0..1 scale, never a sum.
    st.bar_chart(df.set_index("model")[["accuracy", "cohen_kappa", "hallucination_recall"]],
                 stack=False)
    st.caption("Calibration set: 40 human-labeled cases. With n=40, a 0.05 accuracy gap is "
               "two items; check the McNemar p-values in the script output before ranking.")

# ------------------------------------------------------------ Human labeling
elif page == "Human labeling":
    st.title("Human evaluation")
    if runs.empty:
        st.info("No runs yet.")
        st.stop()
    run_id = st.selectbox("Run", runs["run_id"])
    res = load_results(run_id)
    annotator = st.text_input("Annotator", value="me")
    done = {l["item_id"] for l in store.human_labels(run_id) if l["annotator"] == annotator}
    todo = [i for i in sorted(res["item_id"].unique()) if i not in done]
    st.progress(1 - len(todo) / max(res["item_id"].nunique(), 1),
                text=f"{len(done)} labeled, {len(todo)} remaining")
    if todo:
        item = st.selectbox("Item", todo)
        row = res[res["item_id"] == item].iloc[0]
        st.markdown(f"**Question.** {row['question']}")
        st.markdown(f"**Answer.** {row['answer']}")
        with st.expander("Contexts (judge the answer against these)", expanded=True):
            for c in json.loads(row["contexts_json"]):
                st.text(c)
        st.caption("The reference answer and judge scores are hidden on purpose: "
                   "seeing them anchors human judgment.")
        with st.form("label"):
            faithful = st.radio("Every claim supported by the contexts?", [1, 0],
                                format_func=lambda v: "Yes, faithful" if v else "No, unfaithful")
            score = st.slider("Overall quality (1-5)", 1, 5, 3)
            note = st.text_area("Note (what is wrong?)")
            if st.form_submit_button("Save label"):
                store.add_human_label(run_id, item, annotator, faithful, score, note)
                st.success("saved")
                st.rerun()

    labels = pd.DataFrame(store.human_labels(run_id))
    judged = (res[res["metric"] == "faithfulness"][["item_id", "score"]]
              .rename(columns={"score": "judge_score"}))  # labels also have a 'score' column
    if not labels.empty and not judged.empty:
        m = labels.merge(judged, on="item_id")
        m = m[m["judge_score"].notna()]
        if len(m) >= 5:
            agree = binary_agreement(m["faithful"].tolist(),
                                     (m["judge_score"] >= 0.5).astype(int).tolist())
            st.subheader("Judge vs your labels (RAGAS faithfulness >= 0.5 as 'faithful')")
            st.write(agree.as_dict(), f"kappa: {kappa_label(agree.cohen_kappa)}")

# ----------------------------------------------------------------------- Ask
elif page == "Ask":
    st.title("Ask the Helios assistant")
    q = st.text_input("Question", "How long are soft-deleted objects kept on Business?")
    if st.button("Ask"):
        from evallab.rag import build_default_pipeline

        with st.spinner(f"{settings.generator_model} is thinking..."):
            r = build_default_pipeline().answer(q)
        st.markdown(r.answer)
        st.dataframe(pd.DataFrame({"source": r.sources, "score": r.retrieval_scores}))
        st.caption(f"retrieval {r.retrieval_latency_s * 1000:.0f} ms, "
                   f"generation {r.generation_latency_s:.1f} s")
