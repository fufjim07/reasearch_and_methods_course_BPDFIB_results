import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind, pearsonr
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from synthetic_date import data, sss_columns

# -------------------------
# 0. Prepare variables
# -------------------------

data["bpd_dummy"] = (data["group"] == "BPD").astype(int)
data["female_dummy"] = (data["gender"] == "Female").astype(int)

sss_columns = [
    "sss_fatigue",
    "sss_unrefreshed_sleep",
    "sss_cognitive_symptoms",
    "sss_headache",
    "sss_abdominal_pain",
    "sss_depression"
]

data["sss_total"] = data[sss_columns].sum(axis=1)

# -------------------------
# 1. Descriptive statistics
# -------------------------

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
print(pd.crosstab(data["group"], data["fm_diagnosis"], normalize="index") * 100)

# -------------------------
# 2. Group matching tests
# -------------------------

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

# Education by group, optional
edu_bpd = data.loc[data["group"] == "BPD", "education_years"]
edu_control = data.loc[data["group"] == "Control", "education_years"]

t_edu, p_edu = ttest_ind(edu_bpd, edu_control, equal_var=True)

print("\nEducation by Group:")
print(data.groupby("group")["education_years"].agg(["mean", "std"]))
print(f"t = {t_edu:.2f}, p = {p_edu:.3f}")

# -------------------------
# 3. Correlation table
# -------------------------

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
for var1 in continuous_vars:
    for var2 in continuous_vars:
        if var1 != var2:
            r, p = pearsonr(data[var1], data[var2])
            print(f"{var1} - {var2}: r = {r:.2f}, p = {p:.3f}")

# -------------------------
# 4. Chi-square: group x FM
# -------------------------

fm_group_table = pd.crosstab(data["group"], data["fm_diagnosis"])
chi2_fm, p_fm, dof_fm, expected_fm = chi2_contingency(fm_group_table)

n = fm_group_table.to_numpy().sum()
cramers_v = np.sqrt(chi2_fm / (n * (min(fm_group_table.shape) - 1)))

print("\nGroup x FM diagnosis:")
print(fm_group_table)
print(f"chi2 = {chi2_fm:.2f}, df = {dof_fm}, p = {p_fm:.3f}")
print(f"Cramer's V = {cramers_v:.3f}")

# -------------------------
# 5. Logistic regression
# -------------------------

model = smf.logit(
    "fm_diagnosis ~ bpd_dummy + female_dummy + age",
    data=data
).fit()

print("\nLogistic regression summary:")
print(model.summary())

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

print("\nOdds ratios:")
print(or_table)

# Nagelkerke R2
ll_null = model.llnull
ll_model = model.llf
n = data.shape[0]

r2_cs = 1 - np.exp((2 / n) * (ll_null - ll_model))
r2_nagelkerke = r2_cs / (1 - np.exp((2 / n) * ll_null))

print(f"\nModel chi-square = {model.llr:.2f}")
print(f"Model p-value = {model.llr_pvalue:.3f}")
print(f"Nagelkerke R2 = {r2_nagelkerke:.3f}")

# -------------------------
# 6. Graph: FM diagnosis by group
# -------------------------

fm_rates = data.groupby("group")["fm_diagnosis"].mean() * 100

