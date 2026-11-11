from pathlib import Path
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Key progression milestones from our official ablation table
versions = [
    "v0.1 (Baseline)",
    "Day 4 (Structure-Aware)",
    "Day 5 (Base BGE)",
    "Day 8 (Cross-Encoder)",
    "Day 9 (Multi-Query)",
    "Day 11 (Parent-Child)",
    "Day 18 (Fine-Tuned BGE)",
    "Day 19 (LlamaIndex)",
]

hit_at_4 = [52.9, 58.8, 47.1, 52.9, 52.9, 41.2, 47.1, 70.6]
mrr_scores = [0.417, 0.441, 0.289, 0.407, 0.446, 0.324, 0.402, 0.510]
latencies_ms = [2470, 1010, 16.3, 686, 4033, 292.7, 740, 586.5]

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# 1. Plot Hit@4 Recall Progression
fig, ax1 = plt.subplots(figsize=(10, 5))
color = "#1f77b4"
ax1.set_title("DocPilot: Hit@4 Recall Progression Across Milestones", fontsize=14, fontweight="bold", pad=15)
ax1.set_ylabel("Hit@4 Recall (%)", fontsize=12, fontweight="bold", color=color)
bars = ax1.bar(versions, hit_at_4, color=color, alpha=0.85, width=0.55)
ax1.set_ylim(0, 85)
ax1.tick_params(axis="x", rotation=30, labelsize=10)

# Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)

plt.tight_layout()
chart_path_recall = OUTPUT_DIR / "recall_progression.png"
plt.savefig(chart_path_recall, dpi=300)
plt.close()
print(f"✓ Saved: {chart_path_recall}")

# 2. Plot MRR Progression
fig, ax2 = plt.subplots(figsize=(10, 5))
color_mrr = "#2ca02c"
ax2.set_title("DocPilot: Mean Reciprocal Rank (MRR) Progression", fontsize=14, fontweight="bold", pad=15)
ax2.set_ylabel("MRR Score (0.0 to 1.0)", fontsize=12, fontweight="bold", color=color_mrr)
lines = ax2.plot(versions, mrr_scores, marker="o", color=color_mrr, linewidth=2.5, markersize=8)
ax2.set_ylim(0.2, 0.6)
ax2.tick_params(axis="x", rotation=30, labelsize=10)

for x, y in zip(versions, mrr_scores):
    ax2.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontweight="bold")

plt.tight_layout()
chart_path_mrr = OUTPUT_DIR / "mrr_progression.png"
plt.savefig(chart_path_mrr, dpi=300)
plt.close()
print(f"✓ Saved: {chart_path_mrr}")

print("\nAll performance charts generated successfully!")