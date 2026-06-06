import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind
from synthetic_date import data, sss_columns

# Descriptive statistics for continuous variables by group
descriptive_by_group = data.groupby("group")[
    ["age", "education_years", "msi_bpd_total", "wpi_total", "sss_total"]
].agg(["mean", "std", "min", "max"])


print(descriptive_by_group)

# Frequencies for gender and FM by group
print(pd.crosstab(data["group"], data["gender"], margins=True))

print(pd.crosstab(data["group"], data["fm_diagnosis"], normalize="index") * 100)

print(pd.crosstab(data["group"], data["fm_diagnosis"])) 

#check for no intervention of gender between the two groups, so we can use chi-square test to check if there is a significant
gender_table = pd.crosstab(data["group"], data["gender"])
chi2_gender, p_gender, dof_gender, expected_gender = chi2_contingency(gender_table)

print(gender_table)
print(f"Gender x Group: chi2 = {chi2_gender:.2f}, df = {dof_gender}, p = {p_gender:.3f}")
print(expected_gender)
#check for no intervention of age between the two groups, so we can use t-test to check if there is a significant difference in age between the two groups
age_bpd = data.loc[data["group"] == "BPD", "age"]
age_control = data.loc[data["group"] == "Control", "age"]

t_age, p_age = ttest_ind(age_bpd, age_control, equal_var=True)

print(f"Age by group: t = {t_age:.2f}, p = {p_age:.3f}")
print(data.groupby("group")["age"].agg(["mean", "std"]))

#check for no intervention of education years between the two groups, so we can use t-test to check if there is a significant difference in education years between the two groups
edu_bpd = data.loc[data["group"] == "BPD", "education_years"]
edu_control = data.loc[data["group"] == "Control", "education_years"]

t_edu, p_edu = ttest_ind(edu_bpd, edu_control, equal_var=True)

print(f"Education by group: t = {t_edu:.2f}, p = {p_edu:.3f}")
print(data.groupby("group")["education_years"].agg(["mean", "std"]))

