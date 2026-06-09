from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from regression import data


CSV_OUTPUT_DIR = Path("csv_files")
IMAGE_OUTPUT_DIR = Path("images")


correlation_vars = [
    "age",
    "education_years",
    "msi_bpd_total",
    "wpi_total",
    "sss_total",
]

variable_labels = {
    "age": "Age",
    "education_years": "Education years",
    "msi_bpd_total": "MSI-BPD",
    "wpi_total": "WPI",
    "sss_total": "SSS",
}


def p_to_stars(p_value):
    """
    Convert p-value to significance stars.
    """

    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def create_correlation_table(data, variables, labels):
    """
    Create an APA-style Pearson correlation table.
    The table shows r values with significance stars in the lower triangle.
    """

    table = pd.DataFrame(
        "",
        index=[labels[var] for var in variables],
        columns=[labels[var] for var in variables],
    )

    p_table = pd.DataFrame(
        np.nan,
        index=[labels[var] for var in variables],
        columns=[labels[var] for var in variables],
    )

    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i == j:
                table.iloc[i, j] = "1"
            elif i > j:
                clean_data = data[[var1, var2]].dropna()
                r_value, p_value = pearsonr(clean_data[var1], clean_data[var2])
                table.iloc[i, j] = f"{r_value:.2f}{p_to_stars(p_value)}"
                p_table.iloc[i, j] = p_value

    return table, p_table


def save_correlation_table_png(table, filename):
    """
    Save the correlation table as a clean academic-style PNG.
    """

    display_table = table.reset_index()
    display_table.columns = ["Variable", *table.columns]

    n_rows, n_cols = display_table.shape
    fig_width = 9
    fig_height = 2.1 + (n_rows * 0.48)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    ax.text(
        0,
        1.03,
        "Table 2. Pearson Correlations Between Continuous Study Variables",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )

    table_artist = ax.table(
        cellText=display_table.values,
        colLabels=display_table.columns,
        loc="upper left",
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0.19, 1, 0.75],
        colWidths=[0.25, 0.15, 0.18, 0.15, 0.13, 0.14],
    )

    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(11)
    table_artist.scale(1, 1.35)

    for (row, col), cell in table_artist.get_celld().items():
        cell.set_edgecolor("#FFFFFF")
        cell.set_linewidth(0)
        cell.set_facecolor("#FFFFFF")
        cell.PAD = 0.08

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
        0.05,
        "Note. Values represent Pearson correlations. *p < .05, **p < .01, ***p < .001.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        style="italic",
    )

    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_correlation_outputs(
    csv_output_dir=CSV_OUTPUT_DIR,
    image_output_dir=IMAGE_OUTPUT_DIR,
):
    """
    Save the correlation table as CSV, optional Excel, and PNG.
    """

    correlation_table, correlation_p_values = create_correlation_table(
        data=data,
        variables=correlation_vars,
        labels=variable_labels,
    )

    csv_output_dir = Path(csv_output_dir)
    image_output_dir = Path(image_output_dir)
    csv_output_dir.mkdir(exist_ok=True)
    image_output_dir.mkdir(exist_ok=True)

    correlation_table.to_csv(
        csv_output_dir / "table_2_correlation_table.csv",
        encoding="utf-8-sig",
    )
    correlation_p_values.to_csv(
        csv_output_dir / "table_2_correlation_p_values.csv",
        encoding="utf-8-sig",
    )

    try:
        with pd.ExcelWriter(
            csv_output_dir / "table_2_correlations.xlsx",
            engine="openpyxl",
        ) as writer:
            correlation_table.to_excel(writer, sheet_name="Correlation table")
            correlation_p_values.to_excel(writer, sheet_name="P values")
    except ModuleNotFoundError:
        print("Skipped Excel export because openpyxl is not installed.")

    image_path = image_output_dir / "table_2_correlation_table.png"
    save_correlation_table_png(correlation_table, image_path)

    return correlation_table, correlation_p_values, image_path


def print_correlation_table(save=True):
    """
    Print and optionally save the correlation table.
    """

    if save:
        correlation_table, correlation_p_values, image_path = save_correlation_outputs()
    else:
        correlation_table, correlation_p_values = create_correlation_table(
            data=data,
            variables=correlation_vars,
            labels=variable_labels,
        )
        image_path = None

    print("\nTABLE 2")
    print("Pearson correlations between continuous study variables")
    print(correlation_table.to_string())
    print("\nNote. Values represent Pearson correlations. *p < .05, **p < .01, ***p < .001.")

    print("\nP-value table:")
    print(correlation_p_values.round(3).to_string())

    if save:
        print("\nSaved correlation tables:")
        print(f"{CSV_OUTPUT_DIR / 'table_2_correlation_table.csv'}")
        print(f"{CSV_OUTPUT_DIR / 'table_2_correlation_p_values.csv'}")
        print(f"{image_path}")


if __name__ == "__main__":
    print_correlation_table(save=True)
