import numpy as np
import warnings
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2

from regression import data, model as main_model


# -------------------------
# Model 2: Main effects + interaction
# -------------------------

interaction_model = smf.glm(
    formula="fm_diagnosis ~ bpd_dummy * female_dummy + age",
    data=data,
    family=sm.families.Binomial()
).fit()


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def create_interaction_prediction_grid(data, interaction_model):
    """
    Create adjusted predicted probabilities for the BPD x gender interaction.
    """

    mean_age = data["age"].mean()

    grid = pd.DataFrame({
        "bpd_dummy": [0, 1, 0, 1],
        "female_dummy": [0, 0, 1, 1],
        "age": [mean_age] * 4
    })

    grid["group"] = ["Control", "BPD", "Control", "BPD"]
    grid["gender"] = ["Male", "Male", "Female", "Female"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pred = interaction_model.get_prediction(grid).summary_frame(alpha=0.05)

    grid["pred_pct"] = pred["mean"] * 100
    grid["lower_pct"] = pred["mean_ci_lower"] * 100
    grid["upper_pct"] = pred["mean_ci_upper"] * 100

    return grid


def plot_interaction(grid):
    """
    Plot predicted FM diagnosis probabilities by group and gender.
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    male_data = grid[grid["gender"] == "Male"]
    female_data = grid[grid["gender"] == "Female"]

    x = [0, 1]

    ax.plot(
        x,
        male_data["pred_pct"],
        marker="o",
        linewidth=2.5,
        markersize=8,
        label="Male",
        color="#4C78A8"
    )

    ax.plot(
        x,
        female_data["pred_pct"],
        marker="o",
        linewidth=2.5,
        markersize=8,
        label="Female",
        color="#F08AA5"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(["Control", "BPD"])

    ax.set_ylabel("Predicted probability of FM diagnosis")
    ax.set_xlabel("Study group")
    ax.set_title("BPD x Gender interaction predicting FM diagnosis", pad=16)

    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))

    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(title="Gender")

    plt.tight_layout()
    plt.show()


def print_interaction_analysis():
    """
    Print the interaction model results and interpretation.
    """

    print("\nInteraction model:")
    print(interaction_model.summary())

    interaction_p = interaction_model.pvalues["bpd_dummy:female_dummy"]

    print("\nInteraction p-value:")
    print("BPD x Gender interaction p-value:", round(interaction_p, 4))

    if interaction_p < 0.05:
        print("The interaction is significant.")
        print("This means that the effect of BPD on FM diagnosis differs between men and women.")
    else:
        print("The interaction is not significant.")
        print("This means there is no statistical evidence that the effect of BPD on FM diagnosis differs between men and women.")

    lr_stat = 2 * (interaction_model.llf - main_model.llf)
    df_diff = interaction_model.df_model - main_model.df_model
    lr_p = chi2.sf(lr_stat, df_diff)

    print("\nLikelihood ratio test comparing main model vs interaction model:")
    print("LR statistic:", round(lr_stat, 4))
    print("df difference:", int(df_diff))
    print("p-value:", round(lr_p, 4))

    if lr_p < 0.05:
        print("The interaction model fits significantly better than the main effects model.")
    else:
        print("The interaction model does not fit significantly better than the main effects model.")

    grid = create_interaction_prediction_grid(data, interaction_model)
    print("\nAdjusted predicted probabilities:")
    print(grid.round(3))

    return grid


def run_interaction_analysis(show_plot=False):
    """
    Run the interaction analysis. The plot is optional.
    """

    grid = print_interaction_analysis()

    if show_plot:
        plot_interaction(grid)

    return grid


if __name__ == "__main__":
    run_interaction_analysis(show_plot=True)
