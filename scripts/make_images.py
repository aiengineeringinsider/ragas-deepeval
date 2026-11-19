"""Render the three Helios images used in Module 11.

The numbers in the pricing and credit images come from the Helios corpus.
The latency chart is fictional product data that exists only in this image,
so questions about it can ONLY be answered by actually reading the chart.

Run:  python scripts/make_images.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "data" / "images"
OUT.mkdir(parents=True, exist_ok=True)


def pricing_table():
    rows = [["Starter", "$0.021", "1 TB", "$0.05", "None"],
            ["Business", "$0.018", "5 TB", "$0.03", "99.9%"],
            ["Enterprise", "$0.015*", "Contract", "Contract", "99.99%"]]
    cols = ["Plan", "Storage / GB-month", "Free egress / month", "Extra egress / GB", "Uptime SLA"]
    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(11)
    t.scale(1, 1.8)
    for j in range(len(cols)):
        t[0, j].set_facecolor("#0B4DA2")
        t[0, j].set_text_props(color="white", weight="bold")
    ax.set_title("Helios Cloud Storage plans (*typical list price)", fontsize=12, pad=6)
    fig.savefig(OUT / "pricing_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def latency_chart():
    regions = ["us-east", "us-west", "eu-central", "ap-southeast", "ap-south"]
    p95 = [42, 47, 51, 68, 74]  # fictional, image-only data
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(regions, p95, color=["#1F6FEB"] * 3 + ["#7C3AED"] * 2)
    ax.bar_label(bars, labels=[f"{v} ms" for v in p95], padding=3)
    ax.set_ylabel("p95 GET latency (ms)")
    ax.set_ylim(0, 90)
    ax.set_title("Helios p95 GET latency by region, October 2026")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "latency_by_region.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def credit_chart():
    labels = ["99.0 to <99.9", "95.0 to <99.0", "<95.0"]
    credits = [10, 25, 50]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bars = ax.barh(labels, credits, color="#0E9488")
    ax.bar_label(bars, labels=[f"{c}% credit" for c in credits], padding=4)
    ax.set_xlabel("Service credit (% of monthly storage charges)")
    ax.set_ylabel("Monthly uptime (%)")
    ax.set_xlim(0, 65)
    ax.invert_yaxis()
    ax.set_title("Business plan SLA credits")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "business_sla_credits.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    pricing_table()
    latency_chart()
    credit_chart()
    print("wrote:", *sorted(p.name for p in OUT.glob("*.png")))
