"""Regenerate Section 3 figures from the unified Llama-3.3-70B gen5_query5 analysis.
Reads only the frozen analysis CSVs; writes PDF+PNG to thesis_figures/ (unified_* names).
Academic style: muted palette, minimal chartjunk, left-aligned panel labels.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "Scripts_code" / "outputs" / "analysis_unified_llama70b_gen5_query5"
FIG = ROOT / "thesis_figures"
FIG.mkdir(exist_ok=True)

C_LABEL = "#3B6CA8"   # label-only
C_SCORE = "#C0603A"   # score-based
C_NEUT = "#6E6E6E"
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 350, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold",
})

def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

asr = pd.read_csv(A / "table_asr_by_condition.csv").set_index("feedback_condition")
qe = pd.read_csv(A / "table_query_efficiency_successes_only.csv")
fc = pd.read_csv(A / "table_failed_checks.csv")
paired = pd.read_csv(A / "table_paired_success.csv").iloc[0]
direct = pd.read_csv(A / "table_directional_success.csv")
cm = pd.read_csv(A / "confidence_movement_candidate_level_unified.csv")
conds = ["label_only", "score_based"]
names = {"label_only": "Label-only", "score_based": "Score-based"}
cols = {"label_only": C_LABEL, "score_based": C_SCORE}

# ---- Figure: main feedback comparison (A ASR w/ Wilson CI, B query efficiency) ----
fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.2, 3.5))
x = np.arange(2)
pct = [asr.loc[c, "percentage"] for c in conds]
lo = [asr.loc[c, "percentage"] - asr.loc[c, "wilson_95_ci_low_percentage"] for c in conds]
hi = [asr.loc[c, "wilson_95_ci_high_percentage"] - asr.loc[c, "percentage"] for c in conds]
axA.bar(x, pct, color=[cols[c] for c in conds], width=0.6,
        yerr=[lo, hi], capsize=5, ecolor=C_NEUT)
for xi, c in zip(x, conds):
    n = int(asr.loc[c, "successes"]); axA.text(xi, pct[conds.index(c)] + max(hi) + 0.8,
        f"{pct[conds.index(c)]:.1f}%\n({n}/389)", ha="center", va="bottom", fontsize=8.5)
axA.set_xticks(x); axA.set_xticklabels([names[c] for c in conds])
axA.set_ylabel("Attack success rate (%)"); axA.set_ylim(0, 36)
axA.set_title("A. Attack success rate", loc="left")

qmean = {r["feedback_condition"]: r for _, r in qe[qe["metric"] == "queries_used"].iterrows()}
means = [qmean[c]["mean"] for c in conds]
q25 = [qmean[c]["mean"] - qmean[c]["q25"] for c in conds]
q75 = [qmean[c]["q75"] - qmean[c]["mean"] for c in conds]
axB.bar(x, means, color=[cols[c] for c in conds], width=0.6,
        yerr=[q25, q75], capsize=5, ecolor=C_NEUT)
for xi, c in zip(x, conds):
    axB.text(xi, means[conds.index(c)] + 0.05, f"{means[conds.index(c)]:.2f}",
             ha="center", va="bottom", fontsize=8.5)
axB.set_xticks(x); axB.set_xticklabels([names[c] for c in conds])
axB.set_ylabel("Classifier queries to first success"); axB.set_ylim(0, 3.2)
axB.set_title("B. Query efficiency (successes only)", loc="left")
fig.tight_layout()
save(fig, "fig_unified_feedback_comparison")

# ---- Figure: ASR by original class ----
fig, ax = plt.subplots(figsize=(6.4, 3.6))
classes = ["deceptive", "truthful"]
w = 0.36
for i, c in enumerate(conds):
    vals = [float(direct[(direct.feedback_condition == c) & (direct.gold_label_name == g)]["percentage"].iloc[0]) for g in classes]
    ns = [(int(direct[(direct.feedback_condition == c) & (direct.gold_label_name == g)]["successes"].iloc[0]),
           int(direct[(direct.feedback_condition == c) & (direct.gold_label_name == g)]["total"].iloc[0])) for g in classes]
    xpos = np.arange(len(classes)) + (i - 0.5) * w
    ax.bar(xpos, vals, width=w, color=cols[c], label=names[c])
    for xp, v, (s, t) in zip(xpos, vals, ns):
        ax.text(xp, v + 0.8, f"{v:.1f}%\n({s}/{t})", ha="center", va="bottom", fontsize=8)
ax.set_xticks(np.arange(len(classes))); ax.set_xticklabels([c.capitalize() for c in classes])
ax.set_ylabel("Attack success rate (%)"); ax.set_ylim(0, 52)
ax.set_xlabel("Original predicted class"); ax.legend(frameon=False)
ax.set_title("Attack success rate by original class", loc="left")
fig.tight_layout()
save(fig, "fig_unified_asr_by_class")

# ---- Figure: failed validity checks (individual, overall) ----
ind = fc[(fc.feedback_condition == "overall") & (fc.count_type == "individual_check")]
ind = ind.sort_values("count", ascending=True)
label_map = {"negation_consistency": "Negation consistency", "sbert_similarity": "SBERT similarity",
             "dates_consistency": "Date consistency", "duplicate_candidate": "Duplicate candidate",
             "numbers_consistency": "Number consistency", "not_identical": "Identical to original"}
fig, ax = plt.subplots(figsize=(6.6, 3.4))
ax.barh([label_map.get(s, s) for s in ind.failed_check], ind["count"], color=C_NEUT)
for y, v in enumerate(ind["count"]):
    ax.text(v + 3, y, str(int(v)), va="center", fontsize=8.5)
ax.set_xlabel("Rejected candidates (count)"); ax.set_xlim(0, max(ind["count"]) * 1.12)
ax.set_title("Validity-check failures (3,285 generated candidates)", loc="left")
fig.tight_layout()
save(fig, "fig_unified_failed_validity_checks")

# ---- Figure: paired outcomes ----
fig, ax = plt.subplots(figsize=(6.2, 3.4))
cats = ["Both", "Label-only\nonly", "Score-based\nonly", "Neither"]
vals = [int(paired.both_success), int(paired.label_only_only), int(paired.score_based_only), int(paired.neither_success)]
barcols = [C_NEUT, C_LABEL, C_SCORE, "#C9C9C9"]
ax.bar(cats, vals, color=barcols, width=0.66)
for i, v in enumerate(vals):
    ax.text(i, v + 4, str(v), ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Paired examples (of 389)"); ax.set_ylim(0, 310)
ax.set_title("Paired attack outcomes (McNemar exact $p=0.30$)", loc="left")
fig.tight_layout()
save(fig, "fig_unified_paired_outcomes")

# ---- Figure: confidence movement (score-based) ----
sb = cm[cm.feedback_condition == "score_based"]
fail = sb[~sb.pair_success]["movement_toward_flip"]; win = sb[sb.pair_success]["movement_toward_flip"]
fig, ax = plt.subplots(figsize=(6.8, 3.6))
bins = np.linspace(-0.5, 1.0, 31)
ax.hist(fail, bins=bins, density=True, alpha=0.7, color=C_LABEL, label=f"Failed attacks (n={len(fail)}, mean {fail.mean():.3f})")
ax.hist(win, bins=bins, density=True, alpha=0.7, color=C_SCORE, label=f"Successful attacks (n={len(win)}, mean {win.mean():.3f})")
ax.axvline(0, color=C_NEUT, lw=0.8, ls="--")
ax.set_xlabel("Confidence movement toward flip (original-class probability, before $-$ after)")
ax.set_ylabel("Density"); ax.legend(frameon=False, fontsize=8.5)
ax.set_title("Score-based feedback: per-candidate confidence movement", loc="left")
fig.tight_layout()
save(fig, "fig_unified_confidence_movement")

print("Wrote figures to", FIG)
for f in sorted(FIG.glob("fig_unified_*.pdf")):
    print(" -", f.name)
