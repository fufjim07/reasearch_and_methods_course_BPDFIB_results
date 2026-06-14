# Project repository

[GitHub repository](https://github.com/fufjim07/reasearch_and_methods_course_BPDFIB_results.git)

# 1. Data checks

```text
Data source: Imported data from regression.py
Total participants: 120
BPD participants: 60
Control participants: 60

Gender counts:
gender
Male      60
Female    60

FM diagnosed participants: 18

Missing values in central variables:
group              0
age                0
education_years    0
gender             0
msi_bpd_total      0
wpi_total          0
sss_total          0
fm_diagnosis       0
bpd_dummy          0
female_dummy       0

Basic ranges:
        age  education_years  msi_bpd_total  wpi_total  sss_total
min   18.00            12.00           0.00       0.00       1.00
max   64.00            20.00          10.00       9.00      11.00
mean  34.49            14.36           5.23       4.12       5.40
std   12.07             1.85           3.53       2.08       2.11
```

# 2. Table 1: Descriptive statistics by study group

| Variable | BPD | Control |
| --- | --- | --- |
| N | 60 | 60 |
| Age, M(SD) | 34.73 (12.06) | 34.25 (12.18) |
| Male, n(%) | 30 (50.0%) | 30 (50.0%) |
| Female, n(%) | 30 (50.0%) | 30 (50.0%) |
| FM diagnosis, n(%) | 15 (25.0%) | 3 (5.0%) |
| WPI total, M(SD) | 4.95 (2.00) | 3.28 (1.81) |
| SSS total, M(SD) | 6.75 (1.62) | 4.05 (1.62) |

# 2b. Table 1b: Additional descriptive statistics by study group

| Variable | BPD | Control |
| --- | --- | --- |
| Education years, M(SD) | 14.58 (1.92) | 14.13 (1.76) |
| MSI-BPD total, M(SD) | 8.50 (1.27) | 1.97 (1.37) |

# 3. Group equivalence tests

- Gender by group: chi-square(1) = 0.00, p = 1.000
- Age by group: t(118) = 0.22, p = 0.827
- Education years by group: t(118) = 1.34, p = 0.184

| Test | Statistic | Value | df | p-value |
| --- | --- | --- | --- | --- |
| Gender by group | chi-square | 0.0 | 1 | 1.000 |
| Age by group | t | 0.218 | 118 | 0.827 |
| Education years by group | t | 1.336 | 118 | 0.184 |

# 4. Table 2: Pearson correlations between continuous study variables

| index | Age | Education years | MSI-BPD | WPI | SSS |
| --- | --- | --- | --- | --- | --- |
| Age | 1 |  |  |  |  |
| Education years | 0.17 | 1 |  |  |  |
| MSI-BPD | 0.00 | 0.15 | 1 |  |  |
| WPI | 0.37*** | 0.25** | 0.38*** | 1 |  |
| SSS | 0.16 | 0.14 | 0.63*** | 0.40*** | 1 |

## P-value table

| index | Age | Education years | MSI-BPD | WPI | SSS |
| --- | --- | --- | --- | --- | --- |
| Age |  |  |  |  |  |
| Education years | 0.057 |  |  |  |  |
| MSI-BPD | 0.994 | 0.104 |  |  |  |
| WPI | < .001 | 0.007 | < .001 |  |  |
| SSS | 0.087 | 0.128 | < .001 | < .001 |  |

Note. Values represent Pearson correlations. *p < .05, **p < .01, ***p < .001.

# 5. FM diagnosis prevalence by study group

- BPD: 15/60 = 25.0%
- Control: 3/60 = 5.0%
- chi-square(1) = 7.91, p = 0.005, Cramer's V = 0.257

# 6. Table 3: Logistic regression predicting FM diagnosis

| Predictor | B | SE | z | df | p-value | Odds Ratio | 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BPD diagnosis | 2.071 | 0.713 | 2.904 | 1 | 0.004 | 7.931 | [1.961, 32.084] |
| Gender: female | 1.25 | 0.625 | 2.002 | 1 | 0.045 | 3.492 | [1.027, 11.874] |
| Age | 0.077 | 0.025 | 3.076 | 1 | 0.002 | 1.08 | [1.028, 1.135] |

## Model fit statistics

| Statistic | Value |
| --- | --- |
| Model chi-square | 23.594 |
| Model df | 3 |
| Model p-value | < .001 |
| Nagelkerke R2 | 0.313 |

# 7. Adjusted predicted probabilities

| Comparison | Level | Predicted probability | Percentage | 95% CI lower | 95% CI upper |
| --- | --- | --- | --- | --- | --- |
| Study group | Control | 0.0507 | 5.1 | 1.8 | 14.1 |
| Study group | BPD | 0.2467 | 24.7 | 16.3 | 36.3 |
| Gender | Male | 0.0913 | 9.1 | 4.4 | 18.4 |
| Gender | Female | 0.2145 | 21.5 | 13.6 | 32.5 |

# 8. Figures and image outputs

```text
Exists: images\descriptives_table.png
Exists: images\table_2_correlation_table.png
Exists: images\logistic_regression_table.png
Missing: images\logistic_predicted_probability_by_group.png
Missing: images\logistic_predicted_probability_by_gender.png
Additional result image: images\additional_descriptives_table.png
Additional result image: images\adjusted_fm_probability_by_gender.png
Additional result image: images\adjusted_fm_probability_by_group.png
Additional result image: images\fm_prevalence_by_gender.png
Additional result image: images\fm_prevalence_by_group.png
```

# Quality checks

```text
All quality checks passed.
```