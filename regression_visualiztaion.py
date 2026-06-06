import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from scipy.special import expit
from matplotlib.ticker import PercentFormatter

# Import the dataset and fitted logistic regression model from your regression module
from regression import data, model


IMAGE_OUTPUT_DIR = "images"


# -------------------------
# Copy data
# -------------------------

data = data.copy()


# -------------------------
# Make sure dummy variables exist
# -------------------------

# bpd_dummy: 0 = Control, 1 = BPD
# female_dummy: 0 = Male, 1 = Female

if "bpd_dummy" not in data.columns:
    data["bpd_dummy"] = (data["group"] == "BPD").astype(int)

if "female_dummy" not in data.columns:
    data["female_dummy"] = (data["gender"] == "Female").astype(int)


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
    Format p-value for graph display.
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


def compute_adjusted_predictions(data, model, variable, values, labels, n_boot=3000, seed=42):
    """
    Compute adjusted predicted probabilities from the existing logistic regression model.

    For each value of the target variable, the function sets that variable
    to the same value for all participants, while keeping the other variables
    as they are.

    This produces adjusted predicted probabilities.
    """

    rng = np.random.default_rng(seed)

    params = model.params
    cov = model.cov_params()

    results = []

    for value, label in zip(values, labels):

        temp = data.copy()
        temp[variable] = value

        # Main predicted probability from the fitted model
        mean_prob = model.predict(temp).mean()

        # Simulate coefficient uncertainty for 95% confidence interval
        boot_params = rng.multivariate_normal(
            mean=params.values,
            cov=cov.values,
            size=n_boot
        )

        boot_means = []

        for beta in boot_params:
            beta_dict = dict(zip(params.index, beta))

            linear_pred = (
                beta_dict["Intercept"]
                + beta_dict["bpd_dummy"] * temp["bpd_dummy"]
                + beta_dict["female_dummy"] * temp["female_dummy"]
                + beta_dict["age"] * temp["age"]
            )

            boot_prob = expit(linear_pred).mean()
            boot_means.append(boot_prob)

        lower = np.percentile(boot_means, 2.5)
        upper = np.percentile(boot_means, 97.5)

        results.append({
            "label": label,
            "mean_pct": mean_prob * 100,
            "lower_pct": lower * 100,
            "upper_pct": upper * 100
        })

    return pd.DataFrame(results)


def plot_adjusted_regression_bar(
    pred_df,
    p_value,
    colors,
    title,
    xlabel,
    filename=None,
    show_plot=True
):
    """
    Plot adjusted predicted probabilities from logistic regression.
    Significance stars are based on the logistic regression p-value.
    """

    x = np.arange(len(pred_df))

    stars = p_to_stars(p_value)
    p_text = format_p_value(p_value)

    fig, ax = plt.subplots(figsize=(8, 6))

    yerr_lower = pred_df["mean_pct"] - pred_df["lower_pct"]
    yerr_upper = pred_df["upper_pct"] - pred_df["mean_pct"]

    ax.bar(
        x,
        pred_df["mean_pct"],
        yerr=[yerr_lower, yerr_upper],
        color=colors,
        edgecolor="black",
        linewidth=1.2,
        width=0.62,
        capsize=6
    )

    # Add percentage labels above bars
    for i, row in pred_df.iterrows():
        ax.text(
            x[i],
            row["upper_pct"] + 1.2,
            f'{row["mean_pct"]:.1f}%',
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

    # Add significance bracket
    y_max = pred_df["upper_pct"].max()
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
    ax.set_xticklabels(pred_df["label"], fontsize=12)

    ax.set_ylabel("Adjusted predicted probability of FM diagnosis (%)", fontsize=13)
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

    ax.text(
        0.5,
        -0.20,
        "Note. Bars represent adjusted predicted probabilities from the logistic regression model. "
        "Error bars represent 95% confidence intervals.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9
    )

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    print(f"\n{title}")
    print(pred_df.round(3))
    print(f"Logistic regression p-value: {p_text}, stars = {stars}")


def run_adjusted_regression_visualizations(show_plots=True):
    """
    Create adjusted predicted probability visualizations from the logistic model.
    """

    image_output_path = Path(IMAGE_OUTPUT_DIR)
    image_output_path.mkdir(exist_ok=True)

    bpd_pred = compute_adjusted_predictions(
        data=data,
        model=model,
        variable="bpd_dummy",
        values=[0, 1],
        labels=["Control", "BPD"]
    )

    plot_adjusted_regression_bar(
        pred_df=bpd_pred,
        p_value=model.pvalues["bpd_dummy"],
        colors=["#8FB3FF", "#D9487F"],
        title="Adjusted predicted probability of FM diagnosis by group",
        xlabel="Study group",
        filename=image_output_path / "adjusted_fm_probability_by_group.png",
        show_plot=show_plots
    )

    gender_pred = compute_adjusted_predictions(
        data=data,
        model=model,
        variable="female_dummy",
        values=[0, 1],
        labels=["Male", "Female"]
    )

    plot_adjusted_regression_bar(
        pred_df=gender_pred,
        p_value=model.pvalues["female_dummy"],
        colors=["#4C78A8", "#F08AA5"],
        title="Adjusted predicted probability of FM diagnosis by gender",
        xlabel="Gender",
        filename=image_output_path / "adjusted_fm_probability_by_gender.png",
        show_plot=show_plots
    )

    print("\nP-values from the logistic regression model:")
    print(f"BPD effect: p = {model.pvalues['bpd_dummy']:.4f}")
    print(f"Gender effect: p = {model.pvalues['female_dummy']:.4f}")
    print(f"Age effect: p = {model.pvalues['age']:.4f}")

    return {
        "bpd_pred": bpd_pred,
        "gender_pred": gender_pred
    }


if __name__ == "__main__":
    run_adjusted_regression_visualizations(show_plots=True)
