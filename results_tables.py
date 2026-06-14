import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from regression import data, model, model_stats


CSV_OUTPUT_DIR = "csv_files"
IMAGE_OUTPUT_DIR = "images"


def format_p_value(p_value):
    """
    Format p-values in a compact APA-style format.
    """

    if p_value < 0.001:
        return "< .001"

    return f"{p_value:.3f}"


def format_mean_sd(series):
    """
    Format a continuous variable as mean and standard deviation.
    """

    return f"{series.mean():.2f} ({series.std():.2f})"


def format_n_percent(count, total):
    """
    Format a frequency as n and percentage.
    """

    percent = (count / total) * 100
    return f"{int(count)} ({percent:.1f}%)"


def create_descriptives_table(data):
    """
    Create a descriptive statistics table by study group.
    """

    rows = []
    group_order = ["BPD", "Control"]

    def group_value(group_name, value_function):
        group_data = data[data["group"] == group_name]
        return value_function(group_data)

    rows.append({
        "Variable": "N",
        **{
            group_name: group_value(group_name, lambda group_data: str(len(group_data)))
            for group_name in group_order
        }
    })

    rows.append({
        "Variable": "Age, M (SD)",
        **{
            group_name: group_value(group_name, lambda group_data: format_mean_sd(group_data["age"]))
            for group_name in group_order
        }
    })

    for gender in ["Male", "Female"]:
        rows.append({
            "Variable": f"{gender}, n (%)",
            **{
                group_name: group_value(
                    group_name,
                    lambda group_data, gender=gender: format_n_percent(
                        (group_data["gender"] == gender).sum(),
                        len(group_data)
                    )
                )
                for group_name in group_order
            }
        })

    rows.append({
        "Variable": "FM diagnosis, n (%)",
        **{
            group_name: group_value(
                group_name,
                lambda group_data: format_n_percent(group_data["fm_diagnosis"].sum(), len(group_data))
            )
            for group_name in group_order
        }
    })

    rows.append({
        "Variable": "WPI total, M (SD)",
        **{
            group_name: group_value(group_name, lambda group_data: format_mean_sd(group_data["wpi_total"]))
            for group_name in group_order
        }
    })

    rows.append({
        "Variable": "SSS total, M (SD)",
        **{
            group_name: group_value(group_name, lambda group_data: format_mean_sd(group_data["sss_total"]))
            for group_name in group_order
        }
    })

    return pd.DataFrame(rows)


def create_additional_descriptives_table(data):
    """
    Create an additional descriptive statistics table by study group.
    """

    rows = []
    group_order = ["BPD", "Control"]

    def group_value(group_name, value_function):
        group_data = data[data["group"] == group_name]
        return value_function(group_data)

    rows.append({
        "Variable": "Education years, M (SD)",
        **{
            group_name: group_value(group_name, lambda group_data: format_mean_sd(group_data["education_years"]))
            for group_name in group_order
        }
    })

    rows.append({
        "Variable": "MSI-BPD total, M (SD)",
        **{
            group_name: group_value(group_name, lambda group_data: format_mean_sd(group_data["msi_bpd_total"]))
            for group_name in group_order
        }
    })

    return pd.DataFrame(rows)


def create_logistic_regression_table(model):
    """
    Create a logistic regression results table for the main predictors.
    """

    predictor_labels = {
        "bpd_dummy": "BPD diagnosis",
        "female_dummy": "Gender: female",
        "age": "Age"
    }

    conf = model.conf_int()
    rows = []

    for predictor, label in predictor_labels.items():
        odds_ratio = np.exp(model.params[predictor])
        ci_lower = np.exp(conf.loc[predictor, 0])
        ci_upper = np.exp(conf.loc[predictor, 1])

        rows.append({
            "Predictor": label,
            "B": round(model.params[predictor], 3),
            "SE": round(model.bse[predictor], 3),
            "z": round(model.tvalues[predictor], 3),
            "df": 1,
            "p-value": format_p_value(model.pvalues[predictor]),
            "Odds Ratio": round(odds_ratio, 3),
            "95% CI": f"[{ci_lower:.3f}, {ci_upper:.3f}]"
        })

    return pd.DataFrame(rows)


def create_model_fit_table(model_stats):
    """
    Create a short model fit summary table for the logistic regression.
    """

    return pd.DataFrame([
        {
            "Statistic": "Model chi-square",
            "Value": round(model_stats["model_chi_square"], 3)
        },
        {
            "Statistic": "Model p-value",
            "Value": format_p_value(model_stats["model_p_value"])
        },
        {
            "Statistic": "Nagelkerke R2",
            "Value": round(model_stats["nagelkerke_r2"], 3)
        }
    ])


def save_academic_table_png(table, title, note, filename, col_widths=None):
    """
    Save a dataframe as a clean academic-style PNG table.
    """

    n_rows, n_cols = table.shape
    fig_width = 10
    fig_height = 1.8 + (n_rows * 0.45)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    ax.text(
        0,
        1.03,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold"
    )

    table_artist = ax.table(
        cellText=table.values,
        colLabels=table.columns,
        loc="upper left",
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0.16, 1, 0.78],
        colWidths=col_widths
    )

    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(10.5)
    table_artist.scale(1, 1.35)

    for (row, col), cell in table_artist.get_celld().items():
        cell.set_edgecolor("#FFFFFF")
        cell.set_linewidth(0)
        cell.set_facecolor("#FFFFFF")
        cell.PAD = 0.06

        if row == 0:
            cell.set_text_props(weight="bold")
            cell.visible_edges = "BT"
            cell.set_edgecolor("#222222")
            cell.set_linewidth(1.2)
        elif row == n_rows:
            cell.visible_edges = "B"
            cell.set_edgecolor("#222222")
            cell.set_linewidth(1.2)

        if col == 0:
            cell.set_text_props(ha="left")

    ax.text(
        0,
        0.04,
        note,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        style="italic"
    )

    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_results_tables_png(output_dir=IMAGE_OUTPUT_DIR):
    """
    Save the two main results tables as academic-style PNG files.
    """

    descriptives_table = create_descriptives_table(data)
    additional_descriptives_table = create_additional_descriptives_table(data)
    logistic_table = create_logistic_regression_table(model)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    save_academic_table_png(
        table=descriptives_table,
        title="Table 1. Descriptive Statistics by Study Group",
        note="Note. Values are presented as M (SD) for continuous variables and n (%) for categorical variables.",
        filename=output_path / "descriptives_table.png",
        col_widths=[0.42, 0.29, 0.29]
    )

    save_academic_table_png(
        table=additional_descriptives_table,
        title="Table 1b. Additional Descriptive Statistics by Study Group",
        note="Note. Values are presented as M (SD).",
        filename=output_path / "additional_descriptives_table.png",
        col_widths=[0.42, 0.29, 0.29]
    )

    save_academic_table_png(
        table=logistic_table,
        title="Table 3. Logistic Regression Predicting FM Diagnosis",
        note="Note. OR = odds ratio; CI = confidence interval. df = 1 for each predictor.",
        filename=output_path / "logistic_regression_table.png",
        col_widths=[0.24, 0.09, 0.09, 0.09, 0.07, 0.10, 0.13, 0.19]
    )

    return descriptives_table, additional_descriptives_table, logistic_table


def save_results_tables(csv_output_dir=CSV_OUTPUT_DIR, image_output_dir=IMAGE_OUTPUT_DIR):
    """
    Save the results tables as CSV and PNG files.
    """

    descriptives_table = create_descriptives_table(data)
    additional_descriptives_table = create_additional_descriptives_table(data)
    logistic_table = create_logistic_regression_table(model)
    model_fit_table = create_model_fit_table(model_stats)
    csv_output_path = Path(csv_output_dir)
    csv_output_path.mkdir(exist_ok=True)

    descriptives_table.to_csv(csv_output_path / "descriptives_table.csv", index=False)
    additional_descriptives_table.to_csv(csv_output_path / "additional_descriptives_table.csv", index=False)
    logistic_table.to_csv(csv_output_path / "logistic_regression_table.csv", index=False)
    model_fit_table.to_csv(csv_output_path / "logistic_model_fit_table.csv", index=False)

    save_results_tables_png(output_dir=image_output_dir)

    return descriptives_table, additional_descriptives_table, logistic_table, model_fit_table


def print_results_tables(save=True):
    """
    Print and optionally save the descriptive and logistic regression tables.
    """

    if save:
        descriptives_table, additional_descriptives_table, logistic_table, model_fit_table = save_results_tables()
    else:
        descriptives_table = create_descriptives_table(data)
        additional_descriptives_table = create_additional_descriptives_table(data)
        logistic_table = create_logistic_regression_table(model)
        model_fit_table = create_model_fit_table(model_stats)

    print("\nTABLE 1")
    print("Descriptive Statistics by Study Group")
    print(descriptives_table.to_string(index=False))
    print("Note. Values are presented as M (SD) for continuous variables and n (%) for categorical variables.")

    print("\nTABLE 1b")
    print("Additional Descriptive Statistics by Study Group")
    print(additional_descriptives_table.to_string(index=False))
    print("Note. Values are presented as M (SD).")

    print("\nTABLE 3")
    print("Logistic Regression Predicting FM Diagnosis")
    print(logistic_table.to_string(index=False))
    print("Note. OR = odds ratio; CI = confidence interval. df = 1 for each predictor.")

    print("\nLogistic regression model fit:")
    print(model_fit_table.to_string(index=False))

    if save:
        print("\nSaved tables:")
        print(f"{CSV_OUTPUT_DIR}/descriptives_table.csv")
        print(f"{CSV_OUTPUT_DIR}/additional_descriptives_table.csv")
        print(f"{CSV_OUTPUT_DIR}/logistic_regression_table.csv")
        print(f"{CSV_OUTPUT_DIR}/logistic_model_fit_table.csv")
        print(f"{IMAGE_OUTPUT_DIR}/descriptives_table.png")
        print(f"{IMAGE_OUTPUT_DIR}/additional_descriptives_table.png")
        print(f"{IMAGE_OUTPUT_DIR}/logistic_regression_table.png")


if __name__ == "__main__":
    print_results_tables(save=True)
