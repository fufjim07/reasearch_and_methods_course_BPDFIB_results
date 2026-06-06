# Synthetic Data and Statistical Analysis: BPD and Fibromyalgia

This project was created as part of a Research Methods course at Bar-Ilan University. It supports a fictional academic research paper examining the association between Borderline Personality Disorder (BPD) and Fibromyalgia (FM).

The dataset used in this project is fully synthetic. It was generated for educational purposes only, in order to practice research design, synthetic data generation, statistical testing, regression analysis, result-table production, and data visualization.

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
├── main.py
├── synthetic_date.py
├── regression.py
├── interaction.py
├── visualization.py
├── regression_visualiztaion.py
├── results_tables.py
├── csv_files/
└── images/
```

## Main Files

`main.py`  
The main execution file. It runs the full workflow: synthetic data checks, statistical analyses, interaction analysis, results tables, and visualizations.

`synthetic_date.py`  
Generates the synthetic dataset, including study group, age, gender, education years, MSI-BPD items, WPI items, SSS items, and FM diagnosis.

`regression.py`  
Prepares the data for analysis, creates dummy variables, runs descriptive statistics and group comparison tests, and fits the main logistic regression model predicting FM diagnosis.

`interaction.py`  
Tests whether the effect of BPD on FM diagnosis differs by gender using an interaction model.

`visualization.py`  
Creates unadjusted FM prevalence plots by study group and by gender.

`regression_visualiztaion.py`  
Creates adjusted predicted probability plots based on the logistic regression model.

`results_tables.py`  
Creates academic-style results tables, including a descriptive statistics table and a logistic regression results table. The tables are saved both as CSV files and as PNG images.

## How to Run

Run the full project:

```powershell
python main.py
```

This command runs:

- Synthetic data checks
- Descriptive statistics
- Group matching tests
- Chi-square and Fisher's exact tests
- Pearson correlations
- Logistic regression
- BPD x gender interaction analysis
- Academic-style result tables
- Visualizations

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

## Outputs

All CSV files are saved in:

```text
csv_files/
```

Main CSV outputs:

- `synthetic_fm_bpd_dataset.csv` - the full synthetic dataset
- `descriptives_table.csv` - descriptive statistics table
- `logistic_regression_table.csv` - logistic regression results table
- `logistic_model_fit_table.csv` - logistic regression model fit statistics

All image files are saved in:

```text
images/
```

Main image outputs:

- `descriptives_table.png` - academic-style descriptive statistics table
- `logistic_regression_table.png` - academic-style logistic regression table
- `fm_prevalence_by_group.png` - FM prevalence by study group
- `fm_prevalence_by_gender.png` - FM prevalence by gender
- `adjusted_fm_probability_by_group.png` - adjusted predicted FM probability by group
- `adjusted_fm_probability_by_gender.png` - adjusted predicted FM probability by gender

## Statistical Analyses

The project includes the following analyses:

- Descriptive statistics by study group
- Group matching tests for age, gender, and education
- Chi-square test and Fisher's exact test for the association between study group and FM diagnosis
- Pearson correlations among continuous variables
- Logistic regression predicting FM diagnosis from BPD group, gender, and age
- BPD x gender interaction analysis
- Adjusted predicted probabilities from the logistic regression model

## Notes

This project uses synthetic data only. The results should not be interpreted as real clinical or scientific evidence.

The file name `synthetic_date.py` was kept from the original project, although it refers to synthetic data rather than dates.

The file name `regression_visualiztaion.py` was also kept from the original project, including the spelling, to avoid breaking existing imports.
