from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

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


def dataframe_to_excel_rows(table):
    """
    Convert a dataframe with meaningful index labels into worksheet rows.
    """

    display_table = table.reset_index()
    display_table.columns = ["Variable", *table.columns]
    return [display_table.columns.tolist(), *display_table.fillna("").values.tolist()]


def cell_to_xml(value):
    """
    Convert a Python value to a minimal XLSX cell representation.
    """

    if value == "":
        return "<c/>"

    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        return f"<c><v>{value}</v></c>"

    return f'<c t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def save_basic_xlsx(filename, sheets):
    """
    Save a small multi-sheet XLSX workbook without optional Excel dependencies.
    """

    sheet_relationships = []
    workbook_sheets = []

    for index, (sheet_name, rows) in enumerate(sheets, start=1):
        sheet_relationships.append(
            f'<Relationship Id="rId{index}" '
            f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
        )
        workbook_sheets.append(
            f'<sheet name="{escape(sheet_name)}" sheetId="{index}" r:id="rId{index}"/>'
        )

    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
    ]
    content_types.extend(
        f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
        f'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for index in range(1, len(sheets) + 1)
    )
    content_types.append("</Types>")

    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{''.join(workbook_sheets)}</sheets>"
        "</workbook>"
    )

    with ZipFile(filename, "w", ZIP_DEFLATED) as workbook:
        workbook.writestr("[Content_Types].xml", "".join(content_types))
        workbook.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        workbook.writestr("xl/workbook.xml", workbook_xml)
        workbook.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f"{''.join(sheet_relationships)}"
            "</Relationships>",
        )

        for index, (_, rows) in enumerate(sheets, start=1):
            sheet_rows = []
            for row_number, row_values in enumerate(rows, start=1):
                cells = "".join(cell_to_xml(value) for value in row_values)
                sheet_rows.append(f'<row r="{row_number}">{cells}</row>')

            sheet_xml = (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                f"<sheetData>{''.join(sheet_rows)}</sheetData>"
                "</worksheet>"
            )
            workbook.writestr(f"xl/worksheets/sheet{index}.xml", sheet_xml)


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
        save_basic_xlsx(
            csv_output_dir / "table_2_correlations.xlsx",
            sheets=[
                ("Correlation table", dataframe_to_excel_rows(correlation_table)),
                ("P values", dataframe_to_excel_rows(correlation_p_values)),
            ],
        )

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
