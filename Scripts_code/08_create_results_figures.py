from __future__ import annotations

"""Create thesis-ready figures from frozen final analysis tables.

The figures are deliberately compact and PDF-first. Section 3 uses one
two-panel primary-result figure; secondary result views are generated as
appendix figures.
"""

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "Scripts_code" / "outputs"
MAIN_ANALYSIS_DIR = OUTPUT_DIR / "analysis_main_gen3_query3"
EXPLORATORY_ANALYSIS_DIR = OUTPUT_DIR / "analysis_exploratory_remaining_gen3_query3"
FIGURE_DIR = PROJECT_ROOT / "thesis_figures"

CONDITIONS = ["label_only", "score_based"]
CONDITION_LABELS = {
    "label_only": "Label-only",
    "score_based": "Score-based",
}
ANALYSIS_LABELS = ["Primary", "Exploratory", "Combined held-out"]
ANALYSIS_DISPLAY = {
    "Primary": "Primary",
    "Exploratory": "Exploratory",
    "Combined held-out": "Combined held-out",
}
HUMAN_CHECK_LABELS = {
    "negation_consistency": "Negation consistency",
    "numbers_consistency": "Numbers consistency",
    "sbert_similarity": "SBERT similarity",
    "dates_consistency": "Dates consistency",
    "duplicate_candidate": "Duplicate candidate",
    "not_identical": "Not identical",
}

COLORS = {
    "ink": "#252525",
    "muted": "#666666",
    "grid": "#E7E2DC",
    "rule": "#AFA8A0",
    "paper": "#FFFFFF",
    "label_only": "#525252",
    "score_based": "#9B6A4A",
    "light_label_only": "#D7D7D7",
    "light_score_based": "#E3CDBF",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def assert_close(
    name: str, actual: float, expected: float, tolerance: float = 1e-9
) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"{name}: expected {expected}, found {actual}")


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires a positive denominator")
    p_hat = successes / total
    denominator = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total)
        / denominator
    )
    return center - margin, center + margin


def get_asr_rows(analysis_dir: Path) -> dict[str, dict[str, float | int]]:
    rows = read_csv_rows(analysis_dir / "table_asr_by_condition.csv")
    return {
        row["feedback_condition"]: {
            "successes": int(row["successes"]),
            "total": int(row["total"]),
            "percentage": float(row["percentage"]),
            "ci_low_percentage": float(row["wilson_95_ci_low_percentage"]),
            "ci_high_percentage": float(row["wilson_95_ci_high_percentage"]),
        }
        for row in rows
    }


def get_combined_asr(
    main_asr: dict[str, dict[str, float | int]],
    exploratory_asr: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float | int]]:
    combined: dict[str, dict[str, float | int]] = {}
    for condition in CONDITIONS:
        successes = int(main_asr[condition]["successes"]) + int(
            exploratory_asr[condition]["successes"]
        )
        total = int(main_asr[condition]["total"]) + int(exploratory_asr[condition]["total"])
        ci_low, ci_high = wilson_interval(successes, total)
        combined[condition] = {
            "successes": successes,
            "total": total,
            "percentage": 100.0 * successes / total,
            "ci_low_percentage": 100.0 * ci_low,
            "ci_high_percentage": 100.0 * ci_high,
        }
    return combined


def get_query_success_rows(analysis_dir: Path) -> dict[str, dict[str, float | int]]:
    rows = read_csv_rows(analysis_dir / "table_query_efficiency_successes_only.csv")
    return {
        row["feedback_condition"]: {
            "count": int(row["count"]),
            "mean": float(row["mean"]),
            "median": float(row["median"]),
            "min": float(row["min"]),
            "max": float(row["max"]),
            "q25": float(row["q25"]),
            "q75": float(row["q75"]),
        }
        for row in rows
        if row["metric"] == "queries_used" and row["feedback_condition"] in CONDITIONS
    }


def get_failed_check_counts(analysis_dir: Path) -> dict[str, int]:
    rows = read_csv_rows(analysis_dir / "table_failed_checks.csv")
    return {
        row["failed_check"]: int(row["count"])
        for row in rows
        if row["feedback_condition"] == "overall"
        and row["count_type"] == "individual_check"
    }


def get_paired_counts(analysis_dir: Path) -> dict[str, int]:
    rows = read_csv_rows(analysis_dir / "table_paired_success.csv")
    if len(rows) != 1:
        raise ValueError(f"Expected one paired-success row in {analysis_dir}")
    row = rows[0]
    return {
        "total_pairs": int(row["total_pairs"]),
        "both_success": int(row["both_success"]),
        "label_only_only": int(row["label_only_only"]),
        "score_based_only": int(row["score_based_only"]),
        "neither_success": int(row["neither_success"]),
    }


def collect_plot_data() -> dict[str, object]:
    main_asr = get_asr_rows(MAIN_ANALYSIS_DIR)
    exploratory_asr = get_asr_rows(EXPLORATORY_ANALYSIS_DIR)
    combined_asr = get_combined_asr(main_asr, exploratory_asr)
    main_queries = get_query_success_rows(MAIN_ANALYSIS_DIR)
    exploratory_queries = get_query_success_rows(EXPLORATORY_ANALYSIS_DIR)
    main_failed = get_failed_check_counts(MAIN_ANALYSIS_DIR)
    exploratory_failed = get_failed_check_counts(EXPLORATORY_ANALYSIS_DIR)
    main_paired = get_paired_counts(MAIN_ANALYSIS_DIR)
    exploratory_paired = get_paired_counts(EXPLORATORY_ANALYSIS_DIR)

    failed_combined = {
        check: main_failed.get(check, 0) + exploratory_failed.get(check, 0)
        for check in HUMAN_CHECK_LABELS
    }
    paired_combined = {
        key: main_paired[key] + exploratory_paired[key]
        for key in ["total_pairs", "both_success", "label_only_only", "score_based_only", "neither_success"]
    }

    return {
        "asr": {
            "Primary": main_asr,
            "Exploratory": exploratory_asr,
            "Combined held-out": combined_asr,
        },
        "queries": {
            "Primary": main_queries,
            "Exploratory": exploratory_queries,
        },
        "failed_checks": failed_combined,
        "paired": {
            "Primary": main_paired,
            "Exploratory": exploratory_paired,
            "Combined held-out": paired_combined,
        },
    }


def verify_plot_data(data: dict[str, object]) -> None:
    asr = data["asr"]  # type: ignore[index]
    expected_asr = {
        ("Primary", "label_only"): (19, 160, 11.875),
        ("Primary", "score_based"): (15, 160, 9.375),
        ("Exploratory", "label_only"): (17, 229, 7.423580786026202),
        ("Exploratory", "score_based"): (20, 229, 8.73362445414847),
        ("Combined held-out", "label_only"): (36, 389, 9.254498714652956),
        ("Combined held-out", "score_based"): (35, 389, 8.997429305912596),
    }
    for (analysis_label, condition), (successes, total, percentage) in expected_asr.items():
        row = asr[analysis_label][condition]  # type: ignore[index]
        if int(row["successes"]) != successes or int(row["total"]) != total:
            raise ValueError(f"Unexpected ASR counts for {analysis_label}/{condition}")
        assert_close(
            f"ASR percentage for {analysis_label}/{condition}",
            float(row["percentage"]),
            percentage,
        )

    queries = data["queries"]["Primary"]  # type: ignore[index]
    assert_close("Primary label-only mean queries", float(queries["label_only"]["mean"]), 1.5789473684210527)
    assert_close("Primary score-based mean queries", float(queries["score_based"]["mean"]), 1.1333333333333333)

    failed_checks = data["failed_checks"]  # type: ignore[assignment]
    expected_failed = {
        "negation_consistency": 354,
        "numbers_consistency": 312,
        "sbert_similarity": 69,
        "dates_consistency": 41,
        "duplicate_candidate": 30,
        "not_identical": 12,
    }
    if failed_checks != expected_failed:
        raise ValueError(f"Unexpected combined failed-check counts: {failed_checks}")

    paired = data["paired"]  # type: ignore[index]
    expected_paired = {
        "Primary": {"label_only_only": 4, "score_based_only": 0},
        "Exploratory": {"label_only_only": 6, "score_based_only": 9},
        "Combined held-out": {"label_only_only": 10, "score_based_only": 9},
    }
    for analysis_label, expected in expected_paired.items():
        for key, value in expected.items():
            if paired[analysis_label][key] != value:
                raise ValueError(f"Unexpected paired count for {analysis_label}/{key}")


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.8,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.3,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.2,
            "figure.dpi": 150,
            "savefig.dpi": 350,
            "savefig.facecolor": COLORS["paper"],
            "figure.facecolor": COLORS["paper"],
            "axes.facecolor": COLORS["paper"],
            "axes.edgecolor": COLORS["ink"],
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.55,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = [FIGURE_DIR / f"{stem}.pdf", FIGURE_DIR / f"{stem}.png"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    return paths


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["left"].set_color(COLORS["ink"])
    ax.spines["bottom"].set_color(COLORS["ink"])
    ax.tick_params(axis="both", colors=COLORS["ink"], length=2.5, width=0.55)
    ax.xaxis.label.set_color(COLORS["ink"])
    ax.yaxis.label.set_color(COLORS["ink"])
    ax.title.set_color(COLORS["ink"])


def add_panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_title(f"{label}. {title}", loc="left", pad=4)


def create_primary_feedback_comparison(data: dict[str, object]) -> list[Path]:
    asr = data["asr"]["Primary"]  # type: ignore[index]
    queries = data["queries"]["Primary"]  # type: ignore[index]

    fig, (ax_asr, ax_query) = plt.subplots(
        1,
        2,
        figsize=(6.15, 2.35),
        gridspec_kw={"width_ratios": [1.1, 1.0], "wspace": 0.35},
        constrained_layout=True,
    )

    y_positions = [1, 0]
    colors = [COLORS["label_only"], COLORS["score_based"]]
    light_colors = [COLORS["light_label_only"], COLORS["light_score_based"]]

    percentages = [float(asr[c]["percentage"]) for c in CONDITIONS]
    ci_low = [float(asr[c]["ci_low_percentage"]) for c in CONDITIONS]
    ci_high = [float(asr[c]["ci_high_percentage"]) for c in CONDITIONS]
    xerr = [
        [percentages[i] - ci_low[i] for i in range(len(CONDITIONS))],
        [ci_high[i] - percentages[i] for i in range(len(CONDITIONS))],
    ]

    ax_asr.barh(y_positions, percentages, color=light_colors, height=0.44, edgecolor="none")
    ax_asr.errorbar(
        percentages,
        y_positions,
        xerr=xerr,
        fmt="o",
        color=COLORS["ink"],
        ecolor=COLORS["rule"],
        elinewidth=0.9,
        capsize=2.2,
        markersize=3.6,
        zorder=3,
    )
    for y, condition, percentage, color in zip(y_positions, CONDITIONS, percentages, colors):
        successes = int(asr[condition]["successes"])
        total = int(asr[condition]["total"])
        ax_asr.text(
            percentage + 0.55,
            y,
            f"{successes}/{total} ({percentage:.1f}%)",
            va="center",
            ha="left",
            color=color,
            fontsize=7.5,
        )
    ax_asr.set_yticks(y_positions)
    ax_asr.set_yticklabels([CONDITION_LABELS[c] for c in CONDITIONS])
    ax_asr.set_xlim(0, 20)
    ax_asr.set_xlabel("Attack success rate (%)")
    ax_asr.grid(axis="x")
    add_panel_label(ax_asr, "A", "Primary attack success")
    clean_axis(ax_asr)

    means = [float(queries[c]["mean"]) for c in CONDITIONS]
    q25 = [float(queries[c]["q25"]) for c in CONDITIONS]
    q75 = [float(queries[c]["q75"]) for c in CONDITIONS]
    min_values = [float(queries[c]["min"]) for c in CONDITIONS]
    max_values = [float(queries[c]["max"]) for c in CONDITIONS]
    for y, min_value, max_value in zip(y_positions, min_values, max_values):
        ax_query.hlines(y, min_value, max_value, color=COLORS["grid"], linewidth=3.0, zorder=1)
    for y, low, high in zip(y_positions, q25, q75):
        ax_query.hlines(
            y,
            low,
            high,
            color=COLORS["rule"],
            linewidth=1.35,
            zorder=2,
        )
        ax_query.vlines(
            [low, high],
            y - 0.055,
            y + 0.055,
            color=COLORS["rule"],
            linewidth=0.9,
            zorder=2,
        )
    ax_query.scatter(
        means,
        y_positions,
        s=18,
        color=COLORS["ink"],
        edgecolor=COLORS["paper"],
        linewidth=0.4,
        zorder=3,
    )
    for y, condition, mean, color in zip(y_positions, CONDITIONS, means, colors):
        count = int(queries[condition]["count"])
        ax_query.text(
            mean + 0.11,
            y,
            f"mean {mean:.2f}; n={count}",
            va="center",
            ha="left",
            color=color,
            fontsize=7.5,
        )
    ax_query.set_yticks(y_positions)
    ax_query.set_yticklabels([CONDITION_LABELS[c] for c in CONDITIONS])
    ax_query.set_xlim(0.8, 3.15)
    ax_query.set_xlabel("Queries to success")
    ax_query.grid(axis="x")
    add_panel_label(ax_query, "B", "Query efficiency")
    clean_axis(ax_query)

    return save_figure(fig, "fig_primary_feedback_comparison")


def create_appendix_asr_summary(data: dict[str, object]) -> list[Path]:
    asr = data["asr"]  # type: ignore[index]
    y_base = list(range(len(ANALYSIS_LABELS)))[::-1]
    offsets = {"label_only": 0.13, "score_based": -0.13}

    fig, ax = plt.subplots(figsize=(5.65, 2.55), constrained_layout=True)
    for idx, analysis_label in enumerate(ANALYSIS_LABELS):
        y = y_base[idx]
        label_value = float(asr[analysis_label]["label_only"]["percentage"])
        score_value = float(asr[analysis_label]["score_based"]["percentage"])
        ax.plot(
            [label_value, score_value],
            [y, y],
            color=COLORS["rule"],
            linewidth=0.85,
            zorder=1,
        )
        for condition in CONDITIONS:
            row = asr[analysis_label][condition]
            value = float(row["percentage"])
            low = float(row["ci_low_percentage"])
            high = float(row["ci_high_percentage"])
            marker = "o" if condition == "label_only" else "s"
            ax.errorbar(
                value,
                y + offsets[condition],
                xerr=[[value - low], [high - value]],
                fmt=marker,
                color=COLORS[condition],
                ecolor=COLORS["rule"],
                elinewidth=0.75,
                capsize=2,
                markersize=3.5,
                zorder=3,
                label=CONDITION_LABELS[condition] if idx == 0 else None,
            )
            ax.text(
                value + 0.35,
                y + offsets[condition],
                f"{value:.1f}%",
                va="center",
                ha="left",
                fontsize=7.1,
                color=COLORS[condition],
            )

    ax.set_yticks(y_base)
    ax.set_yticklabels(
        [
            "Primary (n=160)",
            "Exploratory (n=229)",
            "Combined held-out (n=389)",
        ]
    )
    ax.set_xlim(0, 20)
    ax.set_xlabel("Attack success rate (%)")
    ax.grid(axis="x")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        handletextpad=0.45,
        columnspacing=0.9,
    )
    add_panel_label(ax, "A.1", "Held-out ASR by feedback condition")
    clean_axis(ax)
    return save_figure(fig, "fig_appendix_asr_summary")


def create_appendix_failed_validity_checks(data: dict[str, object]) -> list[Path]:
    failed_checks = data["failed_checks"]  # type: ignore[assignment]
    sorted_items = sorted(failed_checks.items(), key=lambda item: item[1], reverse=True)
    labels = [HUMAN_CHECK_LABELS[check] for check, _ in sorted_items]
    counts = [count for _, count in sorted_items]
    y_positions = list(range(len(labels)))[::-1]

    fig, ax = plt.subplots(figsize=(5.65, 2.8), constrained_layout=True)
    ax.barh(
        y_positions,
        counts,
        color="#D8D3CD",
        edgecolor=COLORS["ink"],
        linewidth=0.35,
        height=0.46,
    )
    for y, count in zip(y_positions, counts):
        ax.text(count + 7, y, f"{count}", va="center", ha="left", fontsize=7.3, color=COLORS["ink"])
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, max(counts) + 55)
    ax.set_xlabel("Rejected candidate count")
    ax.grid(axis="x")
    add_panel_label(ax, "A.2", "Most frequent validity-check failures")
    clean_axis(ax)
    return save_figure(fig, "fig_appendix_failed_validity_checks")


def create_appendix_discordant_paired_outcomes(data: dict[str, object]) -> list[Path]:
    paired = data["paired"]  # type: ignore[index]
    y_positions = list(range(len(ANALYSIS_LABELS)))[::-1]
    label_only_values = [paired[label]["label_only_only"] for label in ANALYSIS_LABELS]
    score_based_values = [paired[label]["score_based_only"] for label in ANALYSIS_LABELS]

    fig, ax = plt.subplots(figsize=(5.65, 2.55), constrained_layout=True)
    ax.axvline(0, color=COLORS["ink"], linewidth=0.65)
    for y, left, right in zip(y_positions, label_only_values, score_based_values):
        ax.barh(y, -left, color=COLORS["light_label_only"], edgecolor=COLORS["ink"], linewidth=0.35, height=0.42)
        ax.barh(y, right, color=COLORS["light_score_based"], edgecolor=COLORS["ink"], linewidth=0.35, height=0.42)
        if left >= 8:
            ax.text(-left + 0.35, y, str(left), va="center", ha="left", fontsize=7.3)
        else:
            ax.text(-left - 0.35 if left else -0.35, y, str(left), va="center", ha="right", fontsize=7.3)
        ax.text(right + 0.35 if right else 0.35, y, str(right), va="center", ha="left", fontsize=7.3)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [
            "Primary (n=160)",
            "Exploratory (n=229)",
            "Combined held-out (n=389)",
        ]
    )
    ax.set_xlim(-11, 11)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{abs(int(value))}"))
    ax.set_xlabel("Number of discordant paired successes")
    ax.grid(axis="x")
    legend_handles = [
        Patch(
            facecolor=COLORS["light_label_only"],
            edgecolor=COLORS["ink"],
            linewidth=0.35,
            label="Label-only only",
        ),
        Patch(
            facecolor=COLORS["light_score_based"],
            edgecolor=COLORS["ink"],
            linewidth=0.35,
            label="Score-based only",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        ncol=2,
        handletextpad=0.45,
        columnspacing=0.9,
    )
    add_panel_label(ax, "A.3", "Discordant paired outcomes")
    clean_axis(ax)
    return save_figure(fig, "fig_appendix_discordant_paired_outcomes")


def write_latex_snippets() -> Path:
    lines = [
        "% Thesis results figures generated by Scripts_code/08_create_results_figures.py",
        "",
        "% Section 3 main-body figure.",
        "\\begin{figure}[htbp]",
        "    \\centering",
        "    \\includegraphics[width=0.86\\textwidth]{thesis_figures/fig_primary_feedback_comparison.pdf}",
        "    \\caption{Primary feedback comparison. Panel A reports attack success rates for the confirmatory primary experiment, with Wilson 95\\% intervals. Panel B reports query efficiency among successful attacks only, with means and interquartile ranges.}",
        "    \\label{fig:primary-feedback-comparison}",
        "\\end{figure}",
        "",
        "% Appendix figures.",
        "\\begin{figure}[htbp]",
        "    \\centering",
        "    \\includegraphics[width=0.82\\textwidth]{thesis_figures/fig_appendix_asr_summary.pdf}",
        "    \\caption{Held-out attack success rates by feedback condition. The primary experiment is confirmatory; exploratory and combined held-out rows are descriptive summaries.}",
        "    \\label{fig:appendix-asr-summary}",
        "\\end{figure}",
        "",
        "\\begin{figure}[htbp]",
        "    \\centering",
        "    \\includegraphics[width=0.82\\textwidth]{thesis_figures/fig_appendix_failed_validity_checks.pdf}",
        "    \\caption{Most frequent validity-check failures across held-out attack runs. Candidates that failed semantic or detail-preservation checks were rejected before classifier querying.}",
        "    \\label{fig:appendix-failed-validity-checks}",
        "\\end{figure}",
        "",
        "\\begin{figure}[htbp]",
        "    \\centering",
        "    \\includegraphics[width=0.82\\textwidth]{thesis_figures/fig_appendix_discordant_paired_outcomes.pdf}",
        "    \\caption{Discordant paired outcomes by analysis set. Bars show cases where only one feedback condition produced a valid successful attack.}",
        "    \\label{fig:appendix-discordant-paired-outcomes}",
        "\\end{figure}",
        "",
    ]
    output_path = FIGURE_DIR / "figures_latex_snippets.tex"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    data = collect_plot_data()
    verify_plot_data(data)
    configure_matplotlib()

    saved_paths: list[Path] = []
    saved_paths.extend(create_primary_feedback_comparison(data))
    saved_paths.extend(create_appendix_asr_summary(data))
    saved_paths.extend(create_appendix_failed_validity_checks(data))
    saved_paths.extend(create_appendix_discordant_paired_outcomes(data))
    saved_paths.append(write_latex_snippets())

    print("Saved thesis figures and LaTeX snippets:")
    for path in saved_paths:
        print(f"- {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
