"""Publication-Grade Performance & Comparison Diagrams for the Safety Study.

Low-RAM rule: Pure Python standard library + matplotlib (no pandas/seaborn).
Generates Figures 1 through 5 at 300 DPI.
"""

import argparse
import collections
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

# Use headless backend for non-interactive rendering
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR, RESULTS_DIR
from src.evaluate import classify_record

# Colorblind-safe palette (Okabe-Ito / Color Universal Design)
CB_BLUE = "#0072B2"
CB_VERMILLION = "#D55E00"
CB_GREEN = "#009E73"
CB_SKY_BLUE = "#56B4E9"
CB_AMBER = "#E69F00"
CB_PURPLE = "#CC79A7"
CB_GRAY = "#7F7F7F"
CB_LIGHT_GRAY = "#E5E7EB"

WRAPPER_ORDER = ["free_text", "strict_json", "json_with_reasoning", "json_with_escalate"]
WRAPPER_LABELS = {
    "free_text": "Free Text\n(Chat)",
    "strict_json": "Strict JSON\n(ok | blocked)",
    "json_with_reasoning": "JSON +\nReasoning",
    "json_with_escalate": "JSON +\nEscalate",
}


def set_academic_style() -> None:
    """Set clean academic publication styling defaults."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelweight": "semibold",
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.0,
        "figure.titlesize": 13.5,
        "figure.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
    })


# ---------------------------------------------------------------------------
# Figure 1: Safety Trade-off by Wrapper (Polished Legend & Headroom)
# ---------------------------------------------------------------------------
def plot_figure_1(
    summary_data: Dict[str, Any],
    output_path: Path,
) -> None:
    """Figure 1: Grouped bar chart comparing Unsafe Refusal Rate vs. Benign Over-Refusal Rate."""
    set_academic_style()
    by_wrapper = summary_data.get("by_wrapper", {})
    total_records = summary_data.get("total_records", 24)
    models_count = len(summary_data.get("by_model_and_wrapper", {}))

    wrappers = WRAPPER_ORDER
    x = range(len(wrappers))
    width = 0.35

    unsafe_rates = []
    unsafe_err_low = []
    unsafe_err_high = []

    safe_rates = []
    safe_err_low = []
    safe_err_high = []

    for w in wrappers:
        stats = by_wrapper.get(w, {})
        u_rate = stats.get("unsafe_refusal_rate", 0.0) * 100.0
        u_ci = stats.get("unsafe_refusal_ci_95", [u_rate / 100.0, u_rate / 100.0])
        unsafe_rates.append(u_rate)
        unsafe_err_low.append(max(0.0, u_rate - (u_ci[0] * 100.0)))
        unsafe_err_high.append(max(0.0, (u_ci[1] * 100.0) - u_rate))

        s_rate = stats.get("safe_over_refusal_rate", 0.0) * 100.0
        s_ci = stats.get("safe_over_refusal_ci_95", [s_rate / 100.0, s_rate / 100.0])
        safe_rates.append(s_rate)
        safe_err_low.append(max(0.0, s_rate - (s_ci[0] * 100.0)))
        safe_err_high.append(max(0.0, (s_ci[1] * 100.0) - s_rate))

    fig, ax = plt.subplots(figsize=(8.8, 5.5), dpi=300)

    rects1 = ax.bar(
        [pos - width / 2 for pos in x],
        unsafe_rates,
        width,
        yerr=[unsafe_err_low, unsafe_err_high],
        capsize=4,
        label="Unsafe Refusal Rate (Safety Recall, %)",
        color=CB_BLUE,
        edgecolor="#222222",
        linewidth=0.8,
        error_kw={"elinewidth": 1.2, "ecolor": "#222222"},
    )

    rects2 = ax.bar(
        [pos + width / 2 for pos in x],
        safe_rates,
        width,
        yerr=[safe_err_low, safe_err_high],
        capsize=4,
        label="Benign Over-Refusal Rate (False Refusal, %)",
        color=CB_AMBER,
        edgecolor="#222222",
        linewidth=0.8,
        error_kw={"elinewidth": 1.2, "ecolor": "#222222"},
    )

    # Direct value labels on bars
    for rect, val in zip(rects1, unsafe_rates):
        h = rect.get_height()
        ax.annotate(
            f"{val:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h + 3),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#111111",
        )

    for rect, val in zip(rects2, safe_rates):
        h = rect.get_height()
        ax.annotate(
            f"{val:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h + 3),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#111111",
        )

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Figure 1: Safety Recall vs. Benign Over-Refusal across Output Wrappers", pad=32)
    fig.text(
        0.5,
        0.915,
        f"XSTest Evaluation Matrix | N = {total_records} Total Calls across {models_count} Distinct Models | Error bars: 95% Bootstrap CI",
        ha="center",
        fontsize=9,
        color="#555555",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels([WRAPPER_LABELS[w] for w in wrappers])
    # Raised headroom to 125 to completely avoid overlap
    ax.set_ylim(0, 125)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    # Legend positioned cleanly above plot area
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=2,
        frameon=True,
        facecolor="#FAFAFA",
        edgecolor="#CCCCCC",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Generated Figure 1 -> {output_path}")


# ---------------------------------------------------------------------------
# Figure 2: Model x Wrapper Heatmap (N/A Masked in Neutral Gray)
# ---------------------------------------------------------------------------
def plot_figure_2(
    summary_data: Dict[str, Any],
    output_path: Path,
) -> None:
    """Figure 2: Side-by-side 2-panel heatmap: Unsafe Refusal Rate and Schema Compliance."""
    set_academic_style()
    by_model = summary_data.get("by_model_and_wrapper", {})
    models = sorted(list(by_model.keys()))
    wrappers = WRAPPER_ORDER

    clean_model_names = []
    for m in models:
        base = m.split("/")[-1].replace(":free", "")
        clean_model_names.append(base)

    mat_unsafe: List[List[float]] = []
    mat_compliance: List[List[float]] = []

    for m in models:
        row_u = []
        row_c = []
        for w in wrappers:
            stats = by_model[m].get(w, {})
            u_rate = stats.get("unsafe_refusal_rate", 0.0) * 100.0
            row_u.append(u_rate)

            comp = stats.get("schema_compliance_rate")
            c_rate = (comp * 100.0) if comp is not None else 0.0
            row_c.append(c_rate)

        mat_unsafe.append(row_u)
        mat_compliance.append(row_c)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.6), dpi=300)

    # Panel A: Unsafe Refusal Rate
    cmap_blue = plt.cm.Blues
    im1 = ax1.imshow(mat_unsafe, cmap=cmap_blue, vmin=0, vmax=100, aspect="auto")
    ax1.set_title("Panel A: Unsafe Refusal Rate (%)", pad=12)
    ax1.set_xticks(range(len(wrappers)))
    ax1.set_xticklabels([WRAPPER_LABELS[w].replace("\n", " ") for w in wrappers], rotation=15, ha="right")
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels(clean_model_names)

    for i in range(len(models)):
        for j in range(len(wrappers)):
            val = mat_unsafe[i][j]
            color = "white" if val > 60 else "black"
            ax1.text(j, i, f"{val:.0f}%", ha="center", va="center", color=color, fontweight="bold", fontsize=10)

    # Panel B: Schema Compliance Rate with N/A cells masked in neutral gray
    cmap_green = plt.cm.Greens
    im2 = ax2.imshow(mat_compliance, cmap=cmap_green, vmin=0, vmax=100, aspect="auto")
    ax2.set_title("Panel B: Schema Compliance Rate (%)", pad=12)
    ax2.set_xticks(range(len(wrappers)))
    ax2.set_xticklabels([WRAPPER_LABELS[w].replace("\n", " ") for w in wrappers], rotation=15, ha="right")
    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels(clean_model_names)

    for i in range(len(models)):
        for j in range(len(wrappers)):
            w_name = wrappers[j]
            if w_name == "free_text":
                # Mask Free Text cell with neutral light gray rectangle
                rect = patches.Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    facecolor="#F3F4F6",
                    edgecolor="#D1D5DB",
                    linewidth=0.8,
                    zorder=2,
                )
                ax2.add_patch(rect)
                ax2.text(j, i, "N/A", ha="center", va="center", color="#4B5563", fontweight="bold", fontsize=9.5, zorder=3)
            else:
                val = mat_compliance[i][j]
                color = "white" if val > 60 else "#111111"
                ax2.text(j, i, f"{val:.0f}%", ha="center", va="center", color=color, fontweight="bold", fontsize=10, zorder=3)

    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=8.5)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=8.5)

    fig.suptitle("Figure 2: Model × Wrapper Performance Matrix", y=0.98, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Generated Figure 2 -> {output_path}")


# ---------------------------------------------------------------------------
# Figure 3: Latency & State Breakdown (Polished Legend & Inner Labels)
# ---------------------------------------------------------------------------
def plot_figure_3(
    records: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Figure 3: Left panel: Response Latency (mean + min/max). Right panel: 100% Stacked Outcomes."""
    set_academic_style()
    wrappers = WRAPPER_ORDER

    # Latencies
    lat_by_w = collections.defaultdict(list)
    for r in records:
        lat_by_w[r.get("wrapper", "")].append(float(r.get("latency_ms", 0.0)))

    means = []
    mins = []
    maxs = []
    err_low = []
    err_high = []

    for w in wrappers:
        lats = lat_by_w.get(w, [0.0])
        avg = sum(lats) / len(lats)
        mn = min(lats)
        mx = max(lats)
        means.append(avg)
        mins.append(mn)
        maxs.append(mx)
        err_low.append(avg - mn)
        err_high.append(mx - avg)

    # State counts
    state_counts = {w: collections.Counter() for w in wrappers}
    for r in records:
        w = r.get("wrapper", "")
        c = classify_record(r)
        decision = c["behavioral_decision"]
        if w != "free_text" and not r.get("json_valid"):
            decision = "invalid_json"
        state_counts[w][decision] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.0), dpi=300)

    # Panel A: Latency
    x = range(len(wrappers))
    bars = ax1.bar(
        x,
        means,
        width=0.45,
        yerr=[err_low, err_high],
        capsize=5,
        color=CB_SKY_BLUE,
        edgecolor="#222222",
        linewidth=0.8,
        error_kw={"elinewidth": 1.2, "ecolor": "#333333"},
    )
    ax1.set_ylabel("Response Latency (ms)")
    ax1.set_title("Panel A: Mean Latency & Range (Min-Max)", pad=14)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([WRAPPER_LABELS[w] for w in wrappers])
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")

    for bar, m_val in zip(bars, means):
        ax1.annotate(
            f"{m_val:.0f} ms",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            color="#111111",
            fontweight="bold",
            fontsize=9,
        )

    # Panel B: 100% Stacked Outcomes
    y_pos = range(len(wrappers))
    complied_pcts = []
    refused_pcts = []
    escalated_pcts = []
    invalid_pcts = []

    for w in wrappers:
        tot = sum(state_counts[w].values()) or 1
        complied_pcts.append((state_counts[w]["complied"] / tot) * 100.0)
        refused_pcts.append((state_counts[w]["refused"] / tot) * 100.0)
        escalated_pcts.append((state_counts[w]["escalated"] / tot) * 100.0)
        invalid_pcts.append((state_counts[w]["invalid_json"] / tot) * 100.0)

    b1 = ax2.barh(y_pos, complied_pcts, color=CB_GREEN, edgecolor="#222222", label="Complied", height=0.52)
    b2 = ax2.barh(y_pos, refused_pcts, left=complied_pcts, color=CB_VERMILLION, edgecolor="#222222", label="Refused", height=0.52)
    left_esc = [c + r for c, r in zip(complied_pcts, refused_pcts)]
    b3 = ax2.barh(y_pos, escalated_pcts, left=left_esc, color=CB_BLUE, edgecolor="#222222", label="Escalated", height=0.52)
    left_inv = [e + l for e, l in zip(escalated_pcts, left_esc)]
    b4 = ax2.barh(y_pos, invalid_pcts, left=left_inv, color=CB_GRAY, edgecolor="#222222", label="Invalid Schema", height=0.52)

    # Print percentage inside segments >= 10%
    for idx, (c_pct, r_pct) in enumerate(zip(complied_pcts, refused_pcts)):
        if c_pct >= 10.0:
            ax2.text(c_pct / 2, idx, f"{c_pct:.0f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=8.5)
        if r_pct >= 10.0:
            ax2.text(c_pct + (r_pct / 2), idx, f"{r_pct:.0f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=8.5)

    ax2.set_xlabel("Proportion of Outcomes (%)")
    ax2.set_title("Panel B: Behavioral Outcome Breakdown (100% Stacked)", pad=28)
    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels([WRAPPER_LABELS[w].replace("\n", " ") for w in wrappers])
    ax2.set_xlim(0, 100)
    ax2.grid(True, linestyle="--", alpha=0.4, axis="x")

    # Legend placed horizontally above the bars
    ax2.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=4,
        frameon=True,
        facecolor="#FAFAFA",
        edgecolor="#CCCCCC",
        fontsize=8.5,
    )

    fig.suptitle("Figure 3: Computational Latency and Outcome Distributions Across Schema Formats", y=0.98, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Generated Figure 3 -> {output_path}")


# ---------------------------------------------------------------------------
# Figure 4: Model A vs. Model B vs. Model C Direct Comparison Chart
# ---------------------------------------------------------------------------
def plot_figure_4(
    summary_data: Dict[str, Any],
    records: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Figure 4: Side-by-side comparison across the 3 evaluated models."""
    set_academic_style()
    by_model = summary_data.get("by_model_and_wrapper", {})
    models = sorted(list(by_model.keys()))

    clean_model_names = [m.split("/")[-1].replace(":free", "") for m in models]

    # Metrics per model
    ft_unsafe_rates = []
    json_unsafe_rates = []
    over_refusal_rates = []
    compliance_rates = []

    mean_lats = []
    min_lats = []
    max_lats = []
    lat_err_low = []
    lat_err_high = []

    for m in models:
        w_dict = by_model[m]
        # (a) Free-text unsafe refusal rate
        ft_u = w_dict.get("free_text", {}).get("unsafe_refusal_rate", 0.0) * 100.0
        ft_unsafe_rates.append(ft_u)

        # (b) JSON-wrapped unsafe refusal rate (average across 3 JSON wrappers)
        json_u_vals = [
            w_dict.get(w, {}).get("unsafe_refusal_rate", 0.0) * 100.0
            for w in ["strict_json", "json_with_reasoning", "json_with_escalate"]
        ]
        json_u = sum(json_u_vals) / len(json_u_vals) if json_u_vals else 0.0
        json_unsafe_rates.append(json_u)

        # (c) Benign over-refusal rate
        safe_vals = [w_dict.get(w, {}).get("safe_over_refusal_rate", 0.0) * 100.0 for w in w_dict]
        s_rate = sum(safe_vals) / len(safe_vals) if safe_vals else 0.0
        over_refusal_rates.append(s_rate)

        # (d) JSON schema compliance rate
        comp_vals = [
            w_dict.get(w, {}).get("schema_compliance_rate", 0.0) * 100.0
            for w in ["strict_json", "json_with_reasoning", "json_with_escalate"]
            if w_dict.get(w, {}).get("schema_compliance_rate") is not None
        ]
        comp_rate = sum(comp_vals) / len(comp_vals) if comp_vals else 100.0
        compliance_rates.append(comp_rate)

        # Latencies
        m_recs = [r for r in records if r.get("model") == m]
        lats = [float(r.get("latency_ms", 0.0)) for r in m_recs]
        if lats:
            avg_l = sum(lats) / len(lats)
            mn_l = min(lats)
            mx_l = max(lats)
        else:
            avg_l, mn_l, mx_l = 0.0, 0.0, 0.0
        mean_lats.append(avg_l)
        min_lats.append(mn_l)
        max_lats.append(mx_l)
        lat_err_low.append(avg_l - mn_l)
        lat_err_high.append(mx_l - avg_l)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2), dpi=300)

    # Panel A: Grouped bar chart
    x = range(len(models))
    bar_w = 0.20

    b1 = ax1.bar([p - 1.5 * bar_w for p in x], ft_unsafe_rates, bar_w, label="Free-Text Unsafe Refusal (%)", color=CB_SKY_BLUE, edgecolor="#222222", linewidth=0.8)
    b2 = ax1.bar([p - 0.5 * bar_w for p in x], json_unsafe_rates, bar_w, label="JSON Unsafe Refusal (%)", color=CB_BLUE, edgecolor="#222222", linewidth=0.8)
    b3 = ax1.bar([p + 0.5 * bar_w for p in x], over_refusal_rates, bar_w, label="Benign Over-Refusal (%)", color=CB_AMBER, edgecolor="#222222", linewidth=0.8)
    b4 = ax1.bar([p + 1.5 * bar_w for p in x], compliance_rates, bar_w, label="JSON Compliance (%)", color=CB_GREEN, edgecolor="#222222", linewidth=0.8)

    # Labels on bars
    for bar_group in [b1, b2, b3, b4]:
        for rect in bar_group:
            h = rect.get_height()
            ax1.annotate(
                f"{h:.0f}%",
                xy=(rect.get_x() + rect.get_width() / 2, h + 2),
                ha="center",
                va="bottom",
                fontsize=8.0,
                fontweight="bold",
                color="#111111",
            )

    ax1.set_ylabel("Percentage (%)")
    ax1.set_title("Panel A: Model Safety & Compliance Comparison", pad=28)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(clean_model_names)
    ax1.set_ylim(0, 125)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        frameon=True,
        facecolor="#FAFAFA",
        edgecolor="#CCCCCC",
        fontsize=8.0,
    )

    # Panel B: Latency Comparison across Models
    b_lat = ax2.bar(
        x,
        mean_lats,
        width=0.42,
        yerr=[lat_err_low, lat_err_high],
        capsize=5,
        color=CB_PURPLE,
        edgecolor="#222222",
        linewidth=0.8,
        error_kw={"elinewidth": 1.2, "ecolor": "#222222"},
    )
    ax2.set_ylabel("Response Latency (ms)")
    ax2.set_title("Panel B: Mean Latency & Range per Model", pad=14)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(clean_model_names)
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")

    for rect, m_val in zip(b_lat, mean_lats):
        ax2.annotate(
            f"{m_val:.0f} ms",
            xy=(rect.get_x() + rect.get_width() / 2, m_val / 2),
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=9.0,
        )

    fig.suptitle("Figure 4: Direct Cross-Model Safety, Schema Compliance & Latency Comparison", y=0.98, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Generated Figure 4 -> {output_path}")


# ---------------------------------------------------------------------------
# Figure 5: End-to-End System Architecture Diagram (Matplotlib Patches)
# ---------------------------------------------------------------------------
def plot_figure_5(output_path: Path) -> None:
    """Figure 5: Horizontal 5-stage system architecture diagram drawn with FancyBboxPatch."""
    set_academic_style()
    fig, ax = plt.subplots(figsize=(13.0, 4.2), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")

    stages = [
        {
            "num": "Stage 1",
            "title": "Stratified XSTest Data",
            "body": "• 100 Safe / 100 Unsafe\n• 18 Prompt Categories\n• Seed 42 Balanced\n• Strict Interleaving",
            "color": "#E6F0FA",
            "border": "#0072B2",
        },
        {
            "num": "Stage 2",
            "title": "4 Schema Wrappers",
            "body": "• free_text (Chat)\n• strict_json (ok|blocked)\n• json_with_reasoning\n• json_with_escalate",
            "color": "#EAF7EE",
            "border": "#009E73",
        },
        {
            "num": "Stage 3",
            "title": "OpenRouter Client",
            "body": "• 3.5s Rate Limiter\n• Auto-Fallback on 429\n• Spend & Call Guards\n• Streaming JSONL",
            "color": "#FFF5E6",
            "border": "#D55E00",
        },
        {
            "num": "Stage 4",
            "title": "Dual Evaluator",
            "body": "• Schema Status Parser\n• Refusal Keyword RegEx\n• Mismatch Detector\n• 1,000x Bootstrap CIs",
            "color": "#F3E8FF",
            "border": "#7B1FA2",
        },
        {
            "num": "Stage 5",
            "title": "Audit & Figures",
            "body": "• summary_metrics.json\n• blind_qa_audit.csv\n• 300 DPI Publication Plots\n• Reproducibility Docs",
            "color": "#FEE2E2",
            "border": "#DC2626",
        },
    ]

    box_w = 16.5
    box_h = 20.0
    gap = 3.8
    start_x = 2.0
    start_y = 5.0

    for idx, stage in enumerate(stages):
        x = start_x + idx * (box_w + gap)
        y = start_y

        # Fancy box
        bbox = patches.FancyBboxPatch(
            (x, y),
            box_w,
            box_h,
            boxstyle="round,pad=0.6,rounding_size=1.2",
            facecolor=stage["color"],
            edgecolor=stage["border"],
            linewidth=1.8,
            zorder=2,
        )
        ax.add_patch(bbox)

        # Header tag
        ax.text(
            x + box_w / 2,
            y + box_h - 2.2,
            stage["num"].upper(),
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=stage["border"],
            zorder=3,
        )

        # Stage title
        ax.text(
            x + box_w / 2,
            y + box_h - 5.0,
            stage["title"],
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="#111111",
            zorder=3,
        )

        # Stage bullet points
        ax.text(
            x + 1.2,
            y + 6.0,
            stage["body"],
            ha="left",
            va="center",
            fontsize=8.0,
            color="#333333",
            linespacing=1.4,
            zorder=3,
        )

        # Directional Arrow between stages
        if idx < len(stages) - 1:
            arrow_x = x + box_w + 0.5
            arrow_y = y + box_h / 2
            arrow = patches.FancyArrowPatch(
                (arrow_x, arrow_y),
                (arrow_x + gap - 1.0, arrow_y),
                arrowstyle="->,head_width=3.5,head_length=5",
                color="#4B5563",
                linewidth=2.0,
                zorder=4,
            )
            ax.add_patch(arrow)

    ax.set_title("Figure 5: End-to-End Experimental Architecture and Research Pipeline", y=0.96, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Generated Figure 5 -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Publication-Grade Performance Diagrams")
    parser.add_argument(
        "--input",
        type=Path,
        default=RESULTS_DIR / "results_free.jsonl",
        help="Path to results JSONL file",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=RESULTS_DIR / "summary_metrics.json",
        help="Path to summary metrics JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURES_DIR,
        help="Directory where output PNG figures will be saved",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load summary metrics
    if not args.metrics.exists():
        raise FileNotFoundError(f"Summary metrics file not found: {args.metrics}. Run src/evaluate.py first.")
    with open(args.metrics, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    # 2. Load raw results
    if not args.input.exists():
        raise FileNotFoundError(f"Results JSONL file not found: {args.input}")
    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Output paths for all 5 figures
    fig1_path = args.output_dir / "fig1_safety_tradeoff_by_wrapper.png"
    fig2_path = args.output_dir / "fig2_model_wrapper_heatmap.png"
    fig3_path = args.output_dir / "fig3_latency_and_tokens.png"
    fig4_path = args.output_dir / "fig4_model_comparison.png"
    fig5_path = args.output_dir / "fig5_architecture_pipeline.png"

    plot_figure_1(summary_data, fig1_path)
    plot_figure_2(summary_data, fig2_path)
    plot_figure_3(records, fig3_path)
    plot_figure_4(summary_data, records, fig4_path)
    plot_figure_5(fig5_path)

    print("\n" + "=" * 65)
    print("ALL 5 FIGURES GENERATED SUCCESSFULLY (300 DPI):")
    print(f"  1. {fig1_path}")
    print(f"  2. {fig2_path}")
    print(f"  3. {fig3_path}")
    print(f"  4. {fig4_path}")
    print(f"  5. {fig5_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
