# Synthetic Data and Statistical Analysis: BPD and Fibromyalgia

This project was created as part of a Research Methods course at Bar-Ilan University. It supports a fictional academic research paper examining the association between Borderline Personality Disorder (BPD) and Fibromyalgia (FM).

The dataset used in this project is fully synthetic. It was generated for educational purposes only, in order to practice research design, synthetic data generation, statistical testing, regression analysis, result-table production, and data visualization.

GitHub repository: [fufjim07/reasearch_and_methods_course_BPDFIB_results](https://github.com/fufjim07/reasearch_and_methods_course_BPDFIB_results.git)

## Research Aim

The main research question is whether BPD diagnosis predicts a higher prevalence of FM compared with a matched control group, while accounting for background variables such as age and gender.

The synthetic sample includes 120 participants:

- 60 participants in the BPD group
- 60 participants in the control group
- Matched age and gender distributions across groups
- FM-related measures based on WPI and SSS scores

## Project Structure

```text
.
|-- main.py
|-- synthetic_date.py
|-- regression.py
|-- interaction.py
|-- visualization.py
|-- regression_visualiztaion.py
|-- corralation_table.py
|-- results_tables.py
|-- submission_output_report.py
|-- csv_files/
|-- images/
`-- submission_output/
```

## Main Files

`main.py`  
Runs the main workflow: synthetic data checks, statistical analyses, interaction analysis, results tables, and visualizations.

`synthetic_date.py`  
Generates the synthetic dataset, including study group, age, gender, education years, MSI-BPD items, WPI items, SSS items, ACR total for compatibility, and FM diagnosis.

`regression.py`  
Prepares the data for analysis, creates dummy variables, runs descriptive statistics and group comparison tests, and fits the main logistic regression model predicting FM diagnosis.

`interaction.py`  
Tests whether the effect of BPD on FM diagnosis differs by gender using an interaction model.

`visualization.py`  
Creates unadjusted FM prevalence plots by study group and by gender.

`regression_visualiztaion.py`  
Creates adjusted predicted probability plots based on the logistic regression model. The filename is kept with its original spelling to avoid breaking existing imports.

`corralation_table.py`  
Creates Table 2, the Pearson correlation table for continuous study variables. The filename is kept with its original spelling to avoid breaking existing project references.

`results_tables.py`  
Creates academic-style results tables, including Table 1, Table 1b, and Table 3. The tables are saved as CSV files and PNG images.

`submission_output_report.py`  
Creates the final submission-oriented output folder with a complete text report, Markdown report, and CSV versions of the submission tables.

## How to Run

Run the full project:

```powershell
python main.py
```

Run only the results tables:

```powershell
python main.py --tables-only
```

Run the project without opening plot windows:

```powershell
python main.py --no-show-plots
```

Run the project without visualizations:

```powershell
python main.py --skip-visualizations
```

Run the project without result tables:

```powershell
python main.py --skip-tables
```

Show the BPD x gender interaction plot:

```powershell
python main.py --interaction-plot
```

Generate the standalone correlation table:

```powershell
python corralation_table.py
```

Generate the standalone results tables:

```powershell
python results_tables.py
```

Generate the submission-ready output report:

```powershell
python submission_output_report.py
```

## Outputs

All general CSV files are saved in:

```text
csv_files/
```

Main CSV outputs:

- `synthetic_fm_bpd_dataset.csv` - the full synthetic dataset
- `descriptives_table.csv` - Table 1, descriptive statistics by study group
- `additional_descriptives_table.csv` - Table 1b, additional descriptive statistics by study group
- `table_2_correlation_table.csv` - Table 2, Pearson correlation table
- `table_2_correlation_p_values.csv` - p-values for Table 2
- `table_2_correlations.xlsx` - Excel workbook with Table 2 and p-values
- `logistic_regression_table.csv` - Table 3, logistic regression results
- `logistic_model_fit_table.csv` - logistic regression model fit statistics

All image files are saved in:

```text
images/
```

Main image outputs:

- `descriptives_table.png` - academic-style Table 1
- `additional_descriptives_table.png` - academic-style Table 1b
- `table_2_correlation_table.png` - academic-style Table 2
- `logistic_regression_table.png` - academic-style Table 3
- `fm_prevalence_by_group.png` - FM prevalence by study group
- `fm_prevalence_by_gender.png` - FM prevalence by gender
- `adjusted_fm_probability_by_group.png` - adjusted predicted FM probability by group
- `adjusted_fm_probability_by_gender.png` - adjusted predicted FM probability by gender

Submission-ready outputs are saved in:

```text
submission_output/
```

Main submission outputs:

- `full_python_output_for_submission.txt` - complete plain-text output for submission, including the GitHub repository link
- `full_python_output_for_submission.md` - complete Markdown output for submission, including the GitHub repository link
- `table_1_descriptives.csv` - Table 1 for submission
- `table_1b_additional_descriptives.csv` - Table 1b for submission
- `table_2_correlations.csv` - Table 2 for submission
- `table_2_p_values.csv` - p-values for Table 2
- `table_3_logistic_regression.csv` - Table 3 for submission
- `model_fit_statistics.csv` - model fit statistics
- `adjusted_predicted_probabilities.csv` - adjusted predicted probabilities

## Statistical Analyses

The project includes the following analyses:

- Descriptive statistics by study group
- Additional descriptive statistics for education years and MSI-BPD total
- Group matching tests for age, gender, and education
- Chi-square test and Fisher's exact test for the association between study group and FM diagnosis
- Pearson correlations among continuous variables: age, education years, MSI-BPD total, WPI total, and SSS total
- Logistic regression predicting FM diagnosis from BPD group, gender, and age
- BPD x gender interaction analysis
- Adjusted predicted probabilities from the logistic regression model

## Current Reporting Tables

- Table 1: Descriptive statistics by study group, including N, age, gender, FM diagnosis, WPI total, and SSS total.
- Table 1b: Additional descriptive statistics by study group, including education years and MSI-BPD total.
- Table 2: Pearson correlations between continuous study variables.
- Table 3: Logistic regression predicting FM diagnosis.

The main logistic regression model is:

```text
fm_diagnosis ~ bpd_dummy + female_dummy + age
```

## Notes

This project uses synthetic data only. The results should not be interpreted as real clinical or scientific evidence.

The file name `synthetic_date.py` was kept from the original project, although it refers to synthetic data rather than dates.

The file name `regression_visualiztaion.py` was kept from the original project, including the spelling, to avoid breaking existing imports.

The file name `corralation_table.py` was also kept with its original spelling to avoid breaking existing project references.
