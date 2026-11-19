"""A complete evaluation loop in 40 lines: dataset -> outputs -> metrics -> report."""
import re
from statistics import mean

golden = [  # question, reference answer, gold source document
    {"q": "How much does Starter cost per GB-month?", "ref": "$0.021", "src": "pricing"},
    {"q": "How long is soft delete on Business?",     "ref": "30 days", "src": "retention"},
    {"q": "Who is the CEO of Helios?",                "ref": None,     "src": None},
]
outputs = [  # what a (hypothetical) system returned for each question
    {"answer": "Starter costs $0.021 per GB-month [1].", "sources": ["pricing", "sla"]},
    {"answer": "Deleted objects are kept for 7 days [1].", "sources": ["sla", "billing"]},
    {"answer": "The CEO is Maria Chen.", "sources": ["pricing"]},
]

def contains_reference(answer, ref):          # reference-based, deterministic
    return float(ref is not None and ref.lower() in answer.lower())

def retrieval_hit(sources, src):              # reference-based, deterministic
    return float(src in sources) if src else float("nan")

def refused(answer):
    return bool(re.search(r"don't have|not (in|mentioned)", answer, re.I))

def refusal_correct(answer, ref):             # behaviour check
    return float(refused(answer) == (ref is None))

rows = []
for g, o in zip(golden, outputs):
    rows.append({
        "q": g["q"][:38],
        "correct": contains_reference(o["answer"], g["ref"]),
        "hit": retrieval_hit(o["sources"], g["src"]),
        "refusal_ok": refusal_correct(o["answer"], g["ref"]),
    })
for r in rows:
    print(f"{r['q']:<40} correct={r['correct']:.0f} hit={r['hit']} refusal_ok={r['refusal_ok']:.0f}")
hits = [r["hit"] for r in rows if r["hit"] == r["hit"]]   # drop NaN
print(f"\naccuracy={mean(r['correct'] for r in rows):.2f}  hit_rate={mean(hits):.2f}  "
      f"refusal_acc={mean(r['refusal_ok'] for r in rows):.2f}")
