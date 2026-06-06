import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from scipy.stats import fisher_exact
from matplotlib.ticker import PercentFormatter

import statsmodels.api as sm
import statsmodels.formula.api as smf

from regression import data, model


IMAGE_OUTPUT_DIR = "images"


# -------------------------
# Copy data to avoid changing the original dataset directly
# -------------------------

data = data.copy()


# -------------------------
# Prepare variables for analysis
# -------------------------

# Create dummy variables if they do not already exist.
# bpd_dummy: 0 = Control, 1 = BPD
# female_dummy: 0 = Male, 1 = Female

if "bpd_dummy" not in data.columns:
    data["bpd_dummy"] = (data["group"] == "BPD").astype(int)

if "female_dummy" not in data.columns:
    data["female_dummy"] = (data["gender"] == "Female").astype(int)


# -------------------------
# Logistic regression model
# -------------------------

# This model predicts FM diagnosis from BPD group, gender, and age.
# Age is included as a control variable.

model = smf.glm(
    formula="fm_diagnosis ~ bpd_dummy + female_dummy + age",
    data=data,
    family=sm.families.Binomial()
).fit()


# -------------------------
# Helper functions
# -------------------------

def p_to_stars(p):
    """
    Convert p-value to significance stars.
    """
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def format_p_value(p):
    """
    Format p-value for display on the graph.
    """
    if p < 0.001:
        return "p < .001"
    else:
        return f"p = {p:.3f}"


def add_significance_bracket(ax, x1, x2, y, h, text):
    """
    Add a significance bracket between two bars.
    """
    ax.plot(
        [x1, x1, x2, x2],
        [y, y + h, y + h, y],
        color="black",
        linewidth=1.8
    )

    ax.text(
        (x1 + x2) / 2,
        y + h + 0.8,
        text,
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold"
    )


def create_prevalence_summary(data, group_col, outcome_col, order):
    """
    Create a summary table with FM cases, total N, and prevalence percentage.
    """
    summary = (
        data.groupby(group_col)[outcome_col]
        .agg(fm_cases="sum", total="count", prevalence="mean")
        .reset_index()
    )

    summary["percent"] = summary["prevalence"] * 100

    summary[group_col] = pd.Categorical(
        summary[group_col],
        categories=order,
        ordered=True
    )

    summary = summary.sort_values(group_col).reset_index(drop=True)

    return summary


def plot_fm_prevalence_bar(
    data,
    group_col,
    outcome_col,
    order,
    colors,
    title,
    xlabel,
    filename=None,
    show_plot=True
):
    """
    Create a bar chart of FM diagnosis prevalence by a categorical variable.
    The graph includes percentages, raw counts, p-value, and significance stars.

    Important:
    The p-value shown on the graph is based on Fisher's exact test.
    This is an unadjusted test of association between the categorical variable and FM diagnosis.
    """

    # Create summary table
    summary = create_prevalence_summary(
        data=data,
        group_col=group_col,
        outcome_col=outcome_col,
        order=order
    )

    # Create 2x2 contingency table
    contingency_table = pd.crosstab(data[group_col], data[outcome_col])
    contingency_table = contingency_table.reindex(order)
    contingency_table = contingency_table.reindex(columns=[0, 1], fill_value=0)

    # Fisher's exact test
    odds_ratio, p_value = fisher_exact(contingency_table)

    stars = p_to_stars(p_value)
    p_text = format_p_value(p_value)

    # Create plot
    x = np.arange(len(summary))

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(
        x,
        summary["percent"],
        color=colors,
        edgecolor="black",
        linewidth=1.2,
        width=0.62
    )

    # Add percentage and raw n above each bar
    for i, row in summary.iterrows():
        ax.text(
            x[i],
            row["percent"] + 1.2,
            f'{row["percent"]:.1f}%\n(n={int(row["fm_cases"])}/{int(row["total"])})',
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

    # Add significance bracket
    y_max = summary["percent"].max()
    bracket_y = y_max + 5

    add_significance_bracket(
        ax=ax,
        x1=x[0],
        x2=x[1],
        y=bracket_y,
        h=1.5,
        text=stars
    )

    # Add p-value text
    ax.text(
        np.mean(x),
        bracket_y + 5,
        p_text,
        ha="center",
        va="bottom",
        fontsize=10
    )

    # Axis formatting
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=12)

    ax.set_ylabel("FM diagnosis prevalence (%)", fontsize=13)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_title(title, fontsize=15, pad=16)

    upper_limit = max(30, np.ceil((bracket_y + 10) / 5) * 5)
    ax.set_ylim(0, upper_limit)

    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))

    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="x", length=0)

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    # Print numerical summary
    print(f"\n{title}")
    print(summary[[group_col, "fm_cases", "total", "percent"]])
    print(f"Fisher's exact test: odds ratio = {odds_ratio:.3f}, {p_text}, stars = {stars}")

    return summary, p_value


def run_prevalence_visualizations(show_plots=True):
    """
    Create and print summaries for the unadjusted FM prevalence visualizations.
    """

    print("\nLogistic regression p-values:")
    print("BPD group p-value:", round(model.pvalues["bpd_dummy"], 4))
    print("Gender p-value:", round(model.pvalues["female_dummy"], 4))
    print("Age p-value:", round(model.pvalues["age"], 4))
    image_output_path = Path(IMAGE_OUTPUT_DIR)
    image_output_path.mkdir(exist_ok=True)

    group_summary, group_p = plot_fm_prevalence_bar(
        data=data,
        group_col="group",
        outcome_col="fm_diagnosis",
        order=["Control", "BPD"],
        colors=["#8FB3FF", "#D9487F"],
        title="Prevalence of FM diagnosis by group",
        xlabel="Study group",
        filename=image_output_path / "fm_prevalence_by_group.png",
        show_plot=show_plots
    )

    gender_summary, gender_p = plot_fm_prevalence_bar(
        data=data,
        group_col="gender",
        outcome_col="fm_diagnosis",
        order=["Male", "Female"],
        colors=["#4C78A8", "#F08AA5"],
        title="Prevalence of FM diagnosis by gender",
        xlabel="Gender",
        filename=image_output_path / "fm_prevalence_by_gender.png",
        show_plot=show_plots
    )

    print("\nFM diagnosis percentages by group:")
    fm_by_group = pd.crosstab(
        data["group"],
        data["fm_diagnosis"],
        normalize="index"
    ) * 100

    fm_by_group = fm_by_group.rename(columns={0: "No FM", 1: "FM"})
    print(fm_by_group.round(2))

    print("\nFM diagnosis percentages by gender:")
    fm_by_gender = pd.crosstab(
        data["gender"],
        data["fm_diagnosis"],
        normalize="index"
    ) * 100

    fm_by_gender = fm_by_gender.rename(columns={0: "No FM", 1: "FM"})
    print(fm_by_gender.round(2))

    print("\nInterpretation of gender effect:")
    print(f"Unadjusted Fisher p-value for gender: {gender_p:.4f}")
    print(f"Adjusted logistic regression p-value for gender: {model.pvalues['female_dummy']:.4f}")

    if gender_p < 0.05:
        print("In the unadjusted Fisher test, gender is significantly associated with FM diagnosis.")
    else:
        print("In the unadjusted Fisher test, gender is not significantly associated with FM diagnosis.")

    if model.pvalues["female_dummy"] < 0.05:
        print("In the logistic regression model, gender is a significant predictor of FM diagnosis after controlling for BPD group and age.")
    else:
        print("In the logistic regression model, gender is not a significant predictor of FM diagnosis after controlling for BPD group and age.")

    return {
        "group_summary": group_summary,
        "group_p": group_p,
        "gender_summary": gender_summary,
        "gender_p": gender_p
    }


if __name__ == "__main__":
    run_prevalence_visualizations(show_plots=True)
