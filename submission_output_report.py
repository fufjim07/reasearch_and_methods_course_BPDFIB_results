from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency, pearsonr, ttest_ind


OUTPUT_DIR = Path("submission_output")
CSV_FALLBACK_PATH = Path("csv_files") / "synthetic_fm_bpd_dataset.csv"
PROJECT_REPOSITORY_URL = "https://github.com/fufjim07/reasearch_and_methods_course_BPDFIB_results.git"


def format_p_value(p_value):
    if p_value < 0.001:
        return "p < .001"
    return f"p = {p_value:.3f}"


def format_p_value_cell(p_value):
    if p_value < 0.001:
        return "< .001"
    return f"{p_value:.3f}"


def format_mean_sd(series):
    return f"{series.mean():.2f} ({series.std():.2f})"


def format_n_percent(count, total):
    percent = (count / total) * 100
    return f"{int(count)} ({percent:.1f}%)"


def p_to_stars(p_value):
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def load_project_data():
    try:
        from regression import data as project_data
        data = project_data.copy()
        source = "Imported data from regression.py"
    except Exception:
        try:
            from synthetic_date import data as project_data
            data = project_data.copy()
            source = "Imported data from synthetic_date.py"
        except Exception:
            data = pd.read_csv(CSV_FALLBACK_PATH)
            source = f"Loaded data from {CSV_FALLBACK_PATH}"

    prepare_required_columns(data)
    return data, source


def prepare_required_columns(data):
    required_columns = [
        "age",
        "education_years",
        "msi_bpd_total",
        "wpi_total",
        "sss_total",
        "fm_diagnosis",
    ]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    if "female_dummy" not in data.columns:
        if "gender" not in data.columns:
            raise ValueError("Missing both female_dummy and gender.")
        data["female_dummy"] = (data["gender"] == "Female").astype(int)

    if "bpd_dummy" not in data.columns:
        if "group" not in data.columns:
            raise ValueError("Missing both bpd_dummy and group.")
        data["bpd_dummy"] = (data["group"] == "BPD").astype(int)

    if "acr_total" not in data.columns:
        data["acr_total"] = data["wpi_total"] + data["sss_total"]


def fit_logistic_model(data):
    return smf.logit(
        "fm_diagnosis ~ bpd_dummy + female_dummy + age",
        data=data,
    ).fit(disp=False)


def create_table_1(data):
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
        },
    })

    rows.append({
        "Variable": "Age, M(SD)",
        **{
            group_name: group_value(group_name, lambda group_data: format_mean_sd(group_data["age"]))
            for group_name in group_order
        },
    })

    for gender in ["Male", "Female"]:
        rows.append({
            "Variable": f"{gender}, n(%)",
            **{
                group_name: group_value(
                    group_name,
                    lambda group_data, gender=gender: format_n_percent(
                        (group_data["gender"] == gender).sum(),
                        len(group_data),
                    ),
                )
                for group_name in group_order
            },
        })

    rows.append({
        "Variable": "FM diagnosis, n(%)",
        **{
            group_name: group_value(
                group_name,
                lambda group_data: format_n_percent(group_data["fm_diagnosis"].sum(), len(group_data)),
            )
            for group_name in group_order
        },
    })

    rows.append({
        "Variable": "WPI total, M(SD)",
        **{
            group_name: group_value(
                group_name,
                lambda group_data: format_mean_sd(group_data["wpi_total"]),
            )
            for group_name in group_order
        },
    })

    rows.append({
        "Variable": "SSS total, M(SD)",
        **{
            group_name: group_value(
                group_name,
                lambda group_data: format_mean_sd(group_data["sss_total"]),
            )
            for group_name in group_order
        },
    })

    return pd.DataFrame(rows)


def create_additional_descriptives_table(data):
    rows = []
    group_order = ["BPD", "Control"]

    def group_value(group_name, value_function):
        group_data = data[data["group"] == group_name]
        return value_function(group_data)

    rows.append({
        "Variable": "Education years, M(SD)",
        **{
            group_name: group_value(
                group_name,
                lambda group_data: format_mean_sd(group_data["education_years"]),
            )
            for group_name in group_order
        },
    })

    rows.append({
        "Variable": "MSI-BPD total, M(SD)",
        **{
            group_name: group_value(
                group_name,
                lambda group_data: format_mean_sd(group_data["msi_bpd_total"]),
            )
            for group_name in group_order
        },
    })

    return pd.DataFrame(rows)


def create_group_equivalence_tests(data):
    gender_table = pd.crosstab(data["group"], data["gender"])
    chi2_gender, p_gender, df_gender, _ = chi2_contingency(gender_table)

    age_bpd = data.loc[data["group"] == "BPD", "age"]
    age_control = data.loc[data["group"] == "Control", "age"]
    t_age, p_age = ttest_ind(age_bpd, age_control, equal_var=True)
    df_age = len(age_bpd) + len(age_control) - 2

    edu_bpd = data.loc[data["group"] == "BPD", "education_years"]
    edu_control = data.loc[data["group"] == "Control", "education_years"]
    t_edu, p_edu = ttest_ind(edu_bpd, edu_control, equal_var=True)
    df_edu = len(edu_bpd) + len(edu_control) - 2

    return pd.DataFrame([
        {
            "Test": "Gender by group",
            "Statistic": "chi-square",
            "Value": round(chi2_gender, 3),
            "df": int(df_gender),
            "p-value": format_p_value_cell(p_gender),
            "Sentence": f"Gender by group: chi-square({df_gender}) = {chi2_gender:.2f}, {format_p_value(p_gender)}",
        },
        {
            "Test": "Age by group",
            "Statistic": "t",
            "Value": round(t_age, 3),
            "df": int(df_age),
            "p-value": format_p_value_cell(p_age),
            "Sentence": f"Age by group: t({df_age}) = {t_age:.2f}, {format_p_value(p_age)}",
        },
        {
            "Test": "Education years by group",
            "Statistic": "t",
            "Value": round(t_edu, 3),
            "df": int(df_edu),
            "p-value": format_p_value_cell(p_edu),
            "Sentence": f"Education years by group: t({df_edu}) = {t_edu:.2f}, {format_p_value(p_edu)}",
        },
    ])


def create_correlation_tables(data):
    correlation_vars = ["age", "education_years", "msi_bpd_total", "wpi_total", "sss_total"]
    labels = {
        "age": "Age",
        "education_years": "Education years",
        "msi_bpd_total": "MSI-BPD",
        "wpi_total": "WPI",
        "sss_total": "SSS",
    }

    table = pd.DataFrame(
        "",
        index=[labels[var] for var in correlation_vars],
        columns=[labels[var] for var in correlation_vars],
    )
    p_table = pd.DataFrame(
        np.nan,
        index=[labels[var] for var in correlation_vars],
        columns=[labels[var] for var in correlation_vars],
    )

    for i, var1 in enumerate(correlation_vars):
        for j, var2 in enumerate(correlation_vars):
            if i == j:
                table.iloc[i, j] = "1"
            elif i > j:
                clean_data = data[[var1, var2]].dropna()
                r_value, p_value = pearsonr(clean_data[var1], clean_data[var2])
                table.iloc[i, j] = f"{r_value:.2f}{p_to_stars(p_value)}"
                p_table.iloc[i, j] = p_value

    return table, p_table


def format_p_value_table(p_table):
    formatted = p_table.copy().astype(object)
    for row_label in formatted.index:
        for column_label in formatted.columns:
            value = formatted.loc[row_label, column_label]
            if pd.isna(value):
                formatted.loc[row_label, column_label] = ""
            elif value < 0.001:
                formatted.loc[row_label, column_label] = "< .001"
            else:
                formatted.loc[row_label, column_label] = f"{value:.3f}"
    return formatted


def create_fm_prevalence(data):
    fm_group_table = pd.crosstab(data["group"], data["fm_diagnosis"])
    fm_group_table = fm_group_table.reindex(["Control", "BPD"])
    fm_group_table = fm_group_table.reindex(columns=[0, 1], fill_value=0)
    chi2_fm, p_fm, df_fm, _ = chi2_contingency(fm_group_table)
    n = fm_group_table.to_numpy().sum()
    cramers_v = np.sqrt(chi2_fm / (n * (min(fm_group_table.shape) - 1)))

    prevalence_rows = []
    for group_name in ["BPD", "Control"]:
        group_data = data[data["group"] == group_name]
        fm_cases = int(group_data["fm_diagnosis"].sum())
        total = len(group_data)
        prevalence_rows.append({
            "Group": group_name,
            "FM cases": fm_cases,
            "Total": total,
            "Percent": round((fm_cases / total) * 100, 1),
        })

    return pd.DataFrame(prevalence_rows), {
        "chi2": chi2_fm,
        "df": df_fm,
        "p": p_fm,
        "cramers_v": cramers_v,
    }


def create_logistic_regression_table(model):
    predictor_labels = {
        "bpd_dummy": "BPD diagnosis",
        "female_dummy": "Gender: female",
        "age": "Age",
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
            "p-value": format_p_value_cell(model.pvalues[predictor]),
            "Odds Ratio": round(odds_ratio, 3),
            "95% CI": f"[{ci_lower:.3f}, {ci_upper:.3f}]",
        })

    return pd.DataFrame(rows)


def create_model_fit_table(model, data):
    ll_null = model.llnull
    ll_model = model.llf
    n = data.shape[0]
    r2_cs = 1 - np.exp((2 / n) * (ll_null - ll_model))
    r2_nagelkerke = r2_cs / (1 - np.exp((2 / n) * ll_null))

    return pd.DataFrame([
        {
            "Statistic": "Model chi-square",
            "Value": round(model.llr, 3),
        },
        {
            "Statistic": "Model df",
            "Value": int(model.df_model),
        },
        {
            "Statistic": "Model p-value",
            "Value": format_p_value_cell(model.llr_pvalue),
        },
        {
            "Statistic": "Nagelkerke R2",
            "Value": round(r2_nagelkerke, 3),
        },
    ])


def logistic_probability(linear_prediction):
    return 1 / (1 + np.exp(-linear_prediction))


def compute_adjusted_predictions(data, model, variable, values, labels, n_simulations=3000, seed=42):
    rng = np.random.default_rng(seed)
    params = model.params
    covariance = model.cov_params()
    simulated_params = rng.multivariate_normal(
        mean=params.values,
        cov=covariance.values,
        size=n_simulations,
    )

    rows = []
    for value, label in zip(values, labels):
        temp = data.copy()
        temp[variable] = value
        mean_probability = model.predict(temp).mean()

        simulated_means = []
        for beta in simulated_params:
            beta_dict = dict(zip(params.index, beta))
            linear_prediction = (
                beta_dict["Intercept"]
                + beta_dict["bpd_dummy"] * temp["bpd_dummy"]
                + beta_dict["female_dummy"] * temp["female_dummy"]
                + beta_dict["age"] * temp["age"]
            )
            simulated_means.append(logistic_probability(linear_prediction).mean())

        lower = np.percentile(simulated_means, 2.5)
        upper = np.percentile(simulated_means, 97.5)

        rows.append({
            "Comparison": variable,
            "Level": label,
            "Predicted probability": round(mean_probability, 4),
            "Percentage": round(mean_probability * 100, 1),
            "95% CI lower": round(lower * 100, 1),
            "95% CI upper": round(upper * 100, 1),
        })

    return pd.DataFrame(rows)


def create_adjusted_predicted_probabilities(data, model):
    group_predictions = compute_adjusted_predictions(
        data=data,
        model=model,
        variable="bpd_dummy",
        values=[0, 1],
        labels=["Control", "BPD"],
        seed=42,
    )
    gender_predictions = compute_adjusted_predictions(
        data=data,
        model=model,
        variable="female_dummy",
        values=[0, 1],
        labels=["Male", "Female"],
        seed=43,
    )

    group_predictions["Comparison"] = "Study group"
    gender_predictions["Comparison"] = "Gender"
    return pd.concat([group_predictions, gender_predictions], ignore_index=True)


def create_data_checks(data, source):
    central_variables = [
        "group",
        "age",
        "education_years",
        "gender",
        "msi_bpd_total",
        "wpi_total",
        "sss_total",
        "fm_diagnosis",
        "bpd_dummy",
        "female_dummy",
    ]
    existing_central_variables = [var for var in central_variables if var in data.columns]
    range_variables = [
        "age",
        "education_years",
        "msi_bpd_total",
        "wpi_total",
        "sss_total",
    ]

    checks = []
    checks.append(f"Data source: {source}")
    checks.append(f"Total participants: {len(data)}")
    checks.append(f"BPD participants: {int((data['bpd_dummy'] == 1).sum())}")
    checks.append(f"Control participants: {int((data['bpd_dummy'] == 0).sum())}")
    checks.append("")
    checks.append("Gender counts:")
    checks.append(data["gender"].value_counts().to_string())
    checks.append("")
    checks.append(f"FM diagnosed participants: {int(data['fm_diagnosis'].sum())}")
    checks.append("")
    checks.append("Missing values in central variables:")
    checks.append(data[existing_central_variables].isna().sum().to_string())
    checks.append("")
    checks.append("Basic ranges:")
    checks.append(data[range_variables].agg(["min", "max", "mean", "std"]).round(2).to_string())
    return "\n".join(checks)


def dataframe_to_markdown(df, include_index=False):
    display_df = df.reset_index() if include_index else df.copy()
    headers = [str(column) for column in display_df.columns]
    rows = display_df.fillna("").astype(str).values.tolist()
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator, *row_lines])


def list_figure_outputs():
    images_dir = Path("images")
    expected_images = [
        images_dir / "descriptives_table.png",
        images_dir / "table_2_correlation_table.png",
        images_dir / "logistic_regression_table.png",
        images_dir / "logistic_predicted_probability_by_group.png",
        images_dir / "logistic_predicted_probability_by_gender.png",
    ]

    lines = []
    reported = set()
    for image_path in expected_images:
        reported.add(image_path)
        if image_path.exists():
            lines.append(f"Exists: {image_path}")
        else:
            lines.append(f"Missing: {image_path}")

    if images_dir.exists():
        for image_path in sorted(images_dir.glob("*.png")):
            if image_path not in reported:
                lines.append(f"Additional result image: {image_path}")

    return "\n".join(lines)


def create_quality_checks(data, table_1, table_1b, correlation_table, table_3):
    warnings = []

    if "acr_total" not in data.columns:
        warnings.append("WARNING: acr_total does not exist.")
    elif not pd.api.types.is_numeric_dtype(data["acr_total"]):
        warnings.append("WARNING: acr_total is not numeric.")
    elif data["acr_total"].nunique(dropna=True) <= 2:
        warnings.append("WARNING: acr_total does not appear continuous.")

    if "fm_diagnosis" not in data.columns:
        warnings.append("WARNING: fm_diagnosis does not exist.")
    else:
        fm_values = set(data["fm_diagnosis"].dropna().unique())
        if not fm_values.issubset({0, 1}):
            warnings.append("WARNING: fm_diagnosis is not dichotomous.")

    table_1_text = " ".join(table_1["Variable"].astype(str))
    if "ACR total" in table_1_text or "acr_total" in table_1_text:
        warnings.append("WARNING: Table 1 includes ACR total or acr_total.")
    if "WPI total" not in table_1_text or "SSS total" not in table_1_text:
        warnings.append("WARNING: Table 1 does not include WPI total and SSS total separately.")

    table_1b_text = " ".join(table_1b["Variable"].astype(str))
    if "Education years" not in table_1b_text or "MSI-BPD total" not in table_1b_text:
        warnings.append("WARNING: Table 1b does not include education_years and msi_bpd_total.")

    table_2_text = " ".join(correlation_table.index.astype(str)) + " " + " ".join(correlation_table.columns.astype(str))
    if "ACR total" in table_2_text or "acr_total" in table_2_text:
        warnings.append("WARNING: Table 2 includes ACR total or acr_total.")
    if "WPI" not in table_2_text or "SSS" not in table_2_text:
        warnings.append("WARNING: Table 2 does not include WPI and SSS separately.")

    table_3_text = " ".join(table_3["Predictor"].astype(str))
    if "acr_total" in table_3_text or "ACR" in table_3_text:
        warnings.append("WARNING: Table 3 includes acr_total.")

    if not warnings:
        return "All quality checks passed."
    return "\n".join(warnings)


def build_report_sections(
    data_checks,
    table_1,
    table_1b,
    equivalence_tests,
    correlation_table,
    p_table,
    prevalence_table,
    prevalence_stats,
    table_3,
    model_fit_table,
    adjusted_predictions,
    figure_outputs,
    quality_checks,
):
    txt_sections = []
    md_sections = []
    p_table_display = format_p_value_table(p_table)

    txt_sections.append("Project repository\n" + PROJECT_REPOSITORY_URL)
    md_sections.append("# Project repository\n\n" + f"[GitHub repository]({PROJECT_REPOSITORY_URL})")

    txt_sections.append("1. Data checks\n" + data_checks)
    md_sections.append("# 1. Data checks\n\n```text\n" + data_checks + "\n```")

    txt_sections.append("2. Table 1: Descriptive statistics by study group\n" + table_1.to_string(index=False))
    md_sections.append("# 2. Table 1: Descriptive statistics by study group\n\n" + dataframe_to_markdown(table_1))

    txt_sections.append(
        "2b. Table 1b: Additional descriptive statistics by study group\n"
        + table_1b.to_string(index=False)
    )
    md_sections.append(
        "# 2b. Table 1b: Additional descriptive statistics by study group\n\n"
        + dataframe_to_markdown(table_1b)
    )

    txt_sections.append(
        "3. Group equivalence tests\n"
        + "\n".join(equivalence_tests["Sentence"].tolist())
        + "\n\n"
        + equivalence_tests.drop(columns=["Sentence"]).to_string(index=False)
    )
    md_sections.append(
        "# 3. Group equivalence tests\n\n"
        + "\n".join(f"- {sentence}" for sentence in equivalence_tests["Sentence"].tolist())
        + "\n\n"
        + dataframe_to_markdown(equivalence_tests.drop(columns=["Sentence"]))
    )

    txt_sections.append(
        "4. Table 2: Pearson correlations between continuous study variables\n"
        + correlation_table.to_string()
        + "\n\nP-value table:\n"
        + p_table_display.to_string()
        + "\n\nNote. Values represent Pearson correlations. *p < .05, **p < .01, ***p < .001."
    )
    md_sections.append(
        "# 4. Table 2: Pearson correlations between continuous study variables\n\n"
        + dataframe_to_markdown(correlation_table, include_index=True)
        + "\n\n## P-value table\n\n"
        + dataframe_to_markdown(p_table_display, include_index=True)
        + "\n\nNote. Values represent Pearson correlations. *p < .05, **p < .01, ***p < .001."
    )

    prevalence_lines = [
        f"{row['Group']}: {int(row['FM cases'])}/{int(row['Total'])} = {row['Percent']:.1f}%"
        for _, row in prevalence_table.iterrows()
    ]
    prevalence_lines.append(
        f"chi-square({prevalence_stats['df']}) = {prevalence_stats['chi2']:.2f}, "
        f"{format_p_value(prevalence_stats['p'])}, "
        f"Cramer's V = {prevalence_stats['cramers_v']:.3f}"
    )
    txt_sections.append("5. FM diagnosis prevalence by study group\n" + "\n".join(prevalence_lines))
    md_sections.append("# 5. FM diagnosis prevalence by study group\n\n" + "\n".join(f"- {line}" for line in prevalence_lines))

    txt_sections.append(
        "6. Table 3: Logistic regression predicting FM diagnosis\n"
        + table_3.to_string(index=False)
        + "\n\nModel fit statistics:\n"
        + model_fit_table.to_string(index=False)
    )
    md_sections.append(
        "# 6. Table 3: Logistic regression predicting FM diagnosis\n\n"
        + dataframe_to_markdown(table_3)
        + "\n\n## Model fit statistics\n\n"
        + dataframe_to_markdown(model_fit_table)
    )

    txt_sections.append(
        "7. Adjusted predicted probabilities\n"
        + adjusted_predictions.to_string(index=False)
    )
    md_sections.append(
        "# 7. Adjusted predicted probabilities\n\n"
        + dataframe_to_markdown(adjusted_predictions)
    )

    txt_sections.append("8. Figures and image outputs\n" + figure_outputs)
    md_sections.append("# 8. Figures and image outputs\n\n```text\n" + figure_outputs + "\n```")

    txt_sections.append("Quality checks\n" + quality_checks)
    md_sections.append("# Quality checks\n\n```text\n" + quality_checks + "\n```")

    separator = "\n\n" + ("=" * 72) + "\n\n"
    return separator.join(txt_sections), "\n\n".join(md_sections)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    data, source = load_project_data()
    model = fit_logistic_model(data)

    data_checks = create_data_checks(data, source)
    table_1 = create_table_1(data)
    table_1b = create_additional_descriptives_table(data)
    equivalence_tests = create_group_equivalence_tests(data)
    correlation_table, p_table = create_correlation_tables(data)
    p_table_display = format_p_value_table(p_table)
    prevalence_table, prevalence_stats = create_fm_prevalence(data)
    table_3 = create_logistic_regression_table(model)
    model_fit_table = create_model_fit_table(model, data)
    adjusted_predictions = create_adjusted_predicted_probabilities(data, model)
    figure_outputs = list_figure_outputs()
    quality_checks = create_quality_checks(data, table_1, table_1b, correlation_table, table_3)

    table_1.to_csv(OUTPUT_DIR / "table_1_descriptives.csv", index=False, encoding="utf-8-sig")
    table_1b.to_csv(OUTPUT_DIR / "table_1b_additional_descriptives.csv", index=False, encoding="utf-8-sig")
    correlation_table.to_csv(OUTPUT_DIR / "table_2_correlations.csv", encoding="utf-8-sig")
    p_table_display.to_csv(OUTPUT_DIR / "table_2_p_values.csv", encoding="utf-8-sig")
    table_3.to_csv(OUTPUT_DIR / "table_3_logistic_regression.csv", index=False, encoding="utf-8-sig")
    model_fit_table.to_csv(OUTPUT_DIR / "model_fit_statistics.csv", index=False, encoding="utf-8-sig")
    adjusted_predictions.to_csv(
        OUTPUT_DIR / "adjusted_predicted_probabilities.csv",
        index=False,
        encoding="utf-8-sig",
    )

    txt_report, md_report = build_report_sections(
        data_checks=data_checks,
        table_1=table_1,
        table_1b=table_1b,
        equivalence_tests=equivalence_tests,
        correlation_table=correlation_table,
        p_table=p_table,
        prevalence_table=prevalence_table,
        prevalence_stats=prevalence_stats,
        table_3=table_3,
        model_fit_table=model_fit_table,
        adjusted_predictions=adjusted_predictions,
        figure_outputs=figure_outputs,
        quality_checks=quality_checks,
    )

    txt_path = OUTPUT_DIR / "full_python_output_for_submission.txt"
    md_path = OUTPUT_DIR / "full_python_output_for_submission.md"
    txt_path.write_text(txt_report, encoding="utf-8")
    md_path.write_text(md_report, encoding="utf-8")

    created_files = [
        txt_path,
        md_path,
        OUTPUT_DIR / "table_1_descriptives.csv",
        OUTPUT_DIR / "table_1b_additional_descriptives.csv",
        OUTPUT_DIR / "table_2_correlations.csv",
        OUTPUT_DIR / "table_2_p_values.csv",
        OUTPUT_DIR / "table_3_logistic_regression.csv",
        OUTPUT_DIR / "model_fit_statistics.csv",
        OUTPUT_DIR / "adjusted_predicted_probabilities.csv",
    ]

    print("Submission output created successfully.")
    print("Created files:")
    for file_path in created_files:
        print(file_path)

    if quality_checks != "All quality checks passed.":
        print("\nQuality check warnings:")
        print(quality_checks)


if __name__ == "__main__":
    main()
