import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind, pearsonr, fisher_exact
import statsmodels.formula.api as smf
from synthetic_date import data as raw_data, sss_columns

# -------------------------
# 0. Prepare data
# -------------------------

def prepare_data(raw_data):
    """
    Prepare the dataset for statistical analyses.
    This function creates dummy variables and recalculates SSS total.
    """

    data = raw_data.copy()

    # bpd_dummy: 0 = Control, 1 = BPD
    data["bpd_dummy"] = (data["group"] == "BPD").astype(int)

    # female_dummy: 0 = Male, 1 = Female
    data["female_dummy"] = (data["gender"] == "Female").astype(int)

    # Recalculate SSS total from the SSS items
    data["sss_total"] = data[sss_columns].sum(axis=1)
    data["acr_total"] = data["wpi_total"] + data["sss_total"]

    return data


data = prepare_data(raw_data)


# -------------------------
# 1. Fit logistic regression model
# -------------------------

def fit_logistic_model(data):
    """
    Fit the main logistic regression model.
    Outcome: FM diagnosis.
    Predictors: BPD group, gender, and age.
    """

    model = smf.logit(
        "fm_diagnosis ~ bpd_dummy + female_dummy + age",
        data=data
    ).fit(disp=False)

    return model


model = fit_logistic_model(data)


# -------------------------
# 2. Create odds ratio table
# -------------------------

def create_odds_ratio_table(model):
    """
    Create odds ratio table for the logistic regression model.
    """

    params = model.params
    conf = model.conf_int()

    or_table = pd.DataFrame({
        "B": params,
        "SE": model.bse,
        "OR": np.exp(params),
        "CI_lower": np.exp(conf[0]),
        "CI_upper": np.exp(conf[1]),
        "p": model.pvalues
    })

    return or_table


or_table = create_odds_ratio_table(model)


# -------------------------
# 3. Model fit statistics
# -------------------------

def calculate_model_fit(model, data):
    """
    Calculate model chi-square, model p-value, and Nagelkerke R2.
    """

    ll_null = model.llnull
    ll_model = model.llf
    n = data.shape[0]

    r2_cs = 1 - np.exp((2 / n) * (ll_null - ll_model))
    r2_nagelkerke = r2_cs / (1 - np.exp((2 / n) * ll_null))

    model_stats = {
        "model_chi_square": model.llr,
        "model_p_value": model.llr_pvalue,
        "nagelkerke_r2": r2_nagelkerke
    }

    return model_stats


model_stats = calculate_model_fit(model, data)


# -------------------------
# 4. Descriptive statistics
# -------------------------

def print_descriptive_statistics(data):
    """
    Print descriptive statistics by group.
    """

    print("\nDescriptive statistics by group:")
    print(
        data.groupby("group")[
            ["age", "education_years", "msi_bpd_total", "wpi_total", "sss_total"]
        ].agg(["mean", "std", "min", "max"])
    )

    print("\nGender by group:")
    print(pd.crosstab(data["group"], data["gender"], margins=True))

    print("\nFM diagnosis by group:")
    print(pd.crosstab(data["group"], data["fm_diagnosis"]))

    print("\nFM diagnosis percentages by group:")
    print(pd.crosstab(data["group"], data["fm_diagnosis"], normalize="index") * 100)

    print("\nFM diagnosis percentages by gender:")
    print(pd.crosstab(data["gender"], data["fm_diagnosis"], normalize="index") * 100)


# -------------------------
# 5. Group matching tests
# -------------------------

def print_group_matching_tests(data):
    """
    Test whether BPD and control groups are matched on gender, age, and education.
    """

    # Gender x group
    gender_table = pd.crosstab(data["group"], data["gender"])
    chi2_gender, p_gender, dof_gender, expected_gender = chi2_contingency(gender_table)

    print("\nGender x Group:")
    print(gender_table)
    print(f"chi2 = {chi2_gender:.2f}, df = {dof_gender}, p = {p_gender:.3f}")

    # Age by group
    age_bpd = data.loc[data["group"] == "BPD", "age"]
    age_control = data.loc[data["group"] == "Control", "age"]

    t_age, p_age = ttest_ind(age_bpd, age_control, equal_var=True)

    print("\nAge by Group:")
    print(data.groupby("group")["age"].agg(["mean", "std"]))
    print(f"t = {t_age:.2f}, p = {p_age:.3f}")

    # Education by group
    edu_bpd = data.loc[data["group"] == "BPD", "education_years"]
    edu_control = data.loc[data["group"] == "Control", "education_years"]

    t_edu, p_edu = ttest_ind(edu_bpd, edu_control, equal_var=True)

    print("\nEducation by Group:")
    print(data.groupby("group")["education_years"].agg(["mean", "std"]))
    print(f"t = {t_edu:.2f}, p = {p_edu:.3f}")


# -------------------------
# 6. Correlation table
# -------------------------

def print_correlation_table(data):
    """
    Print Pearson correlation matrix and p-values for continuous variables.
    """

    continuous_vars = [
        "age",
        "education_years",
        "msi_bpd_total",
        "wpi_total",
        "sss_total"
    ]

    corr_matrix = data[continuous_vars].corr(method="pearson")

    print("\nCorrelation matrix:")
    print(corr_matrix)

    print("\nCorrelation p-values:")
    for i, var1 in enumerate(continuous_vars):
        for var2 in continuous_vars[i + 1:]:
            r, p = pearsonr(data[var1], data[var2])
            print(f"{var1} - {var2}: r = {r:.2f}, p = {p:.3f}")


# -------------------------
# 7. Group x FM diagnosis test
# -------------------------

def print_group_fm_test(data):
    """
    Test the association between study group and FM diagnosis.
    """

    fm_group_table = pd.crosstab(data["group"], data["fm_diagnosis"])
    fm_group_table = fm_group_table.reindex(["Control", "BPD"])
    fm_group_table = fm_group_table.reindex(columns=[0, 1], fill_value=0)

    chi2_fm, p_fm, dof_fm, expected_fm = chi2_contingency(fm_group_table)

    n = fm_group_table.to_numpy().sum()
    cramers_v = np.sqrt(chi2_fm / (n * (min(fm_group_table.shape) - 1)))

    odds_ratio, fisher_p = fisher_exact(fm_group_table)

    print("\nGroup x FM diagnosis:")
    print(fm_group_table)

    print("\nChi-square test:")
    print(f"chi2 = {chi2_fm:.2f}, df = {dof_fm}, p = {p_fm:.3f}")
    print(f"Cramer's V = {cramers_v:.3f}")

    print("\nFisher's exact test:")
    print(f"odds ratio = {odds_ratio:.3f}, p = {fisher_p:.3f}")


# -------------------------
# 8. Logistic regression output
# -------------------------

def print_logistic_regression_results(model, or_table, model_stats):
    """
    Print logistic regression results, odds ratios, and model fit.
    """

    print("\nLogistic regression summary:")
    print(model.summary())

    print("\nOdds ratios:")
    print(or_table)

    print(f"\nModel chi-square = {model_stats['model_chi_square']:.2f}")
    print(f"Model p-value = {model_stats['model_p_value']:.3f}")
    print(f"Nagelkerke R2 = {model_stats['nagelkerke_r2']:.3f}")


# -------------------------
# 9. Optional variables for graphs
# -------------------------

fm_rates = data.groupby("group")["fm_diagnosis"].mean() * 100


# -------------------------
# 10. Run only when this file is executed directly
# -------------------------

def run_all_analyses():
    """
    Run all analyses and print results.
    """

    print_descriptive_statistics(data)
    print_group_matching_tests(data)
    print_correlation_table(data)
    print_group_fm_test(data)
    print_logistic_regression_results(model, or_table, model_stats)


if __name__ == "__main__":
    run_all_analyses()


