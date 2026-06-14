import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42) # For reproducibility of the synthetic dataset

# Step 1: Define the sample size.
# The dataset includes 120 participants.
n_participants = 120
n_per_group = n_participants // 2
participant_id = np.arange(1, n_participants + 1)

# Step 2: Create fixed study groups.
# The sample includes 60 participants with BPD and 60 matched controls.
group = np.array(['BPD'] * n_per_group + ['Control'] * n_per_group)

"""PART 1: Demographic Questionnaire
The demographic questionnaire includes: age, gender, and education.
Neurological or psychiatric conditions that would interfere with the study are excluded!"""
# Step 3: Create gender for the BPD group.
# Gender is balanced within the BPD group.
gender_bpd = np.array(['Male'] * (n_per_group // 2) + ['Female'] * (n_per_group // 2))
np.random.shuffle(gender_bpd)

# Step 4: Match gender in the control group.
# Each control participant receives the same gender as a BPD participant.
gender_control = gender_bpd.copy()

# Step 5: Combine gender for both groups.
gender = np.concatenate([gender_bpd, gender_control])

# Step 6: Create age for the BPD group.
# Ages are generated within the eligible age range of 18 to 65.
# Create age for the BPD group with a broad adult age range.
age_bpd = np.concatenate([
    np.random.normal(loc=25, scale=4, size=25),
    np.random.normal(loc=38, scale=5, size=25),
    np.random.normal(loc=55, scale=5, size=10)
]).astype(int)
age_bpd = np.clip(age_bpd, 18, 65)

# Step 7: Match age in the control group.
# Control ages are generated to be very similar to the BPD participants' ages.
age_control = age_bpd + np.random.randint(-3, 4, size=n_per_group)
age_control = np.clip(age_control, 18, 65)

# Step 8: Combine age for both groups.
age = np.concatenate([age_bpd, age_control])

# Step 9: Create education years.
# Education years are generated similarly in both groups.
# Maximum possible education years for each group
max_education_bpd = age_bpd - 6
max_education_control = age_control - 6

# Education years for BPD group
education_bpd = np.random.normal(loc=14.5, scale=2, size=n_per_group)
education_bpd = np.round(education_bpd).astype(int)
education_bpd = np.clip(education_bpd, 12, 20)
education_bpd = np.minimum(education_bpd, max_education_bpd)
education_bpd = np.clip(education_bpd, 12, 20)

# Education years for control group, NOT matched to BPD group
education_control = np.random.normal(loc=14.5, scale=2, size=n_per_group)
education_control = np.round(education_control).astype(int)
education_control = np.clip(education_control, 12, 20)
education_control = np.minimum(education_control, max_education_control)
education_control = np.clip(education_control, 12, 20)

# Combine education years
education_years = np.concatenate([education_bpd, education_control])

# Step 10: Create the dataframe.
data = pd.DataFrame({
    'id': participant_id,
    'group': group,
    'gender': gender,
    'age': age,
    'education_years': education_years
})


"""PART 2: MSI-BPD Screening Score:
The BPD diagnosis is based on the MSI-BPD, which includes 10 items assessing BPD symptoms.
 Each item is scored as 0 (no) or 1 (yes), and the total score ranges from 0 to 10.
 A score of 7 or higher indicates a positive BPD screening result."""

#step 1: Generate the MSI-BPD items.
# Each item is generated based on the participant's group.
#Assign higher MSI-BPD scores to the BPD group.
# Participants in the BPD group are more likely to answer "yes" to MSI-BPD items.

def generate_msi_bpd_items(group):
    msi_bpd_items = []
    """For control group participants, the probability of endorsing each item is around 20%.
    For BPD group participants, the probability of endorsing each item is around 85%.
    """
    for i in range(1, 11):
        probabilities = np.where(group == 'BPD', 0.85, 0.20)
        item = np.random.binomial(1, probabilities)
        msi_bpd_items.append(item)
    
    # Calculate the total MSI-BPD score for each participant.
    msi_bpd_total = np.sum(msi_bpd_items, axis=0)
    
    return msi_bpd_items, msi_bpd_total


msi_bpd_items, msi_bpd_total = generate_msi_bpd_items(data['group'].values)

# Add the MSI-BPD items to the dataframe.
for i in range(1, 11):
    data[f'msi_bpd_item_{i}'] = msi_bpd_items[i-1]

# Add total MSI-BPD score.
data['msi_bpd_total'] = msi_bpd_total

# Classify positive BPD screening.
data['bpd_screen_positive'] = (data['msi_bpd_total'] >= 7).astype(int)



"""PART 3: Fibromyalgia Diagnosis:
Fibromyalgia is diagnosed based on the 2016 ACR criteria, which include:
1. Widespread Pain Index (WPI): Measures pain in 19 body areas.
2. Symptom Severity Scale (SSS): Measures fatigue, unrefreshing sleep, cognitive
symptoms, headaches, abdominal pain, and depression.
3. Symptom duration: Symptoms must be present for at least three months.
4. Generalized pain: Pain in at least four out of five body regions.
"""
"""PART 3: Fibromyalgia Questionnaire - WPI
The WPI includes 19 body areas.
Each area is scored as 0 = no pain, 1 = pain.
The total WPI score ranges from 0 to 19.
"""

# Names of the 19 WPI pain areas
wpi_areas = [
    "left_shoulder_girdle",
    "right_shoulder_girdle",
    "left_upper_arm",
    "right_upper_arm",
    "left_lower_arm",
    "right_lower_arm",
    "left_jaw",
    "right_jaw",
    "neck",
    "upper_back",
    "lower_back",
    "chest",
    "abdomen",
    "left_hip",
    "right_hip",
    "left_upper_leg",
    "right_upper_leg",
    "left_lower_leg",
    "right_lower_leg"
]

def generate_wpi_items(data):
    wpi_items = []
    
    for area in wpi_areas:
        # Base probability of pain
        probabilities = (
            0.14
            + np.where(data["group"] == "BPD", 0.07, 0.00)
            + np.where(data["gender"] == "Female", 0.05, 0.00)
            + 0.003 * (data["age"] - 30)
        )
        
        # Keep probabilities within a valid range
        probabilities = np.clip(probabilities, 0.03, 0.75)
        
        item = np.random.binomial(1, probabilities)
        wpi_items.append(item)
    
    return wpi_items

# Generate WPI items
wpi_items = generate_wpi_items(data)

# Add WPI items to dataframe
for i, area in enumerate(wpi_areas):
    data[f"wpi_{area}"] = wpi_items[i]

# Calculate WPI total score
wpi_columns = [f"wpi_{area}" for area in wpi_areas]
data["wpi_total"] = data[wpi_columns].sum(axis=1)

"""PART 4: Fibromyalgia Questionnaire - SSS
The SSS includes three severity items scored from 0 to 3:
fatigue, unrefreshed sleep, and cognitive symptoms.
It also includes three binary symptoms:
headache, abdominal pain, and depression.
The total SSS score ranges from 0 to 12.
"""
def generate_severity_item(data, base=0.5):
    latent_score = (
        base
        + np.where(data["group"] == "BPD", 0.45, 0.00)
        + np.where(data["gender"] == "Female", 0.15, 0.00)
        + 0.005 * (data["age"] - 30)
        + 0.08 * data["wpi_total"]
        + np.random.normal(0, 0.85, size=len(data))
    )
    
    score = np.round(latent_score + 0.5).astype(int)
    score = np.clip(score, 0, 3)
    
    return score

# Severity items, each ranges from 0 to 3
data["sss_fatigue"] = generate_severity_item(data, base=0.55)
data["sss_unrefreshed_sleep"] = generate_severity_item(data, base=0.45)
data["sss_cognitive_symptoms"] = generate_severity_item(data, base=0.35)

# Binary SSS symptoms, each ranges from 0 to 1
headache_prob = (
    0.18
    + np.where(data["group"] == "BPD", 0.16, 0.00)
    + np.where(data["gender"] == "Female", 0.12, 0.00)
)
headache_prob = np.clip(headache_prob, 0.05, 0.80)

abdominal_prob = (
    0.14
    + np.where(data["group"] == "BPD", 0.14, 0.00)
    + np.where(data["gender"] == "Female", 0.10, 0.00)
)
abdominal_prob = np.clip(abdominal_prob, 0.05, 0.75)

depression_prob = (
    0.15
    + np.where(data["group"] == "BPD", 0.35, 0.00)
    + np.where(data["gender"] == "Female", 0.05, 0.00)
)
depression_prob = np.clip(depression_prob, 0.05, 0.85)

data["sss_headache"] = np.random.binomial(1, headache_prob)
data["sss_abdominal_pain"] = np.random.binomial(1, abdominal_prob)
data["sss_depression"] = np.random.binomial(1, depression_prob)

# Calculate SSS total
sss_columns = [
    "sss_fatigue",
    "sss_unrefreshed_sleep",
    "sss_cognitive_symptoms",
    "sss_headache",
    "sss_abdominal_pain",
    "sss_depression"
]

data["sss_total"] = data[sss_columns].sum(axis=1)

data["sss_total"] = (
    data["sss_fatigue"]
    + data["sss_unrefreshed_sleep"]
    + data["sss_cognitive_symptoms"]
    + data["sss_headache"]
    + data["sss_abdominal_pain"]
    + data["sss_depression"]
)

data["acr_total"] = data["wpi_total"] + data["sss_total"]

data["fm_diagnosis"] = (
    ((data["wpi_total"] >= 7) & (data["sss_total"] >= 5)) |
    ((data["wpi_total"].between(4, 6)) & (data["sss_total"] >= 9))
).astype(int)

# FM diagnosis percentages by study group
fm_by_group = pd.crosstab(
    data["group"],
    data["fm_diagnosis"],
    normalize="index"
) * 100

fm_by_group = fm_by_group.rename(columns={0: "No FM", 1: "FM"})


# FM diagnosis percentages by gender
fm_by_gender = pd.crosstab(
    data["gender"],
    data["fm_diagnosis"],
    normalize="index"
) * 100

fm_by_gender = fm_by_gender.rename(columns={0: "No FM", 1: "FM"})


def print_synthetic_data_checks():
    """
    Print basic checks for the synthetic dataset.
    """

    print("FM diagnosis percentages by group:")
    print(fm_by_group.round(2))

    print("\nFM diagnosis percentages by gender:")
    print(fm_by_gender.round(2))

    correlation = data["wpi_total"].corr(data["sss_total"])
    print(f"\nCorrelation between WPI total and SSS total: {correlation:.2f}")

    variables = ["age", "education_years", "msi_bpd_total", "wpi_total", "sss_total"]

    print("\nDescriptive statistics:")
    print(data[variables].describe())

    print("\nDescriptive statistics for BPD group:")
    print(data[data["group"] == "BPD"][variables].describe())

    print("\nDescriptive statistics for control group:")
    print(data[data["group"] == "Control"][variables].describe())


def save_synthetic_dataset(filename="synthetic_fm_bpd_dataset.csv", output_dir="csv_files"):
    """
    Save the synthetic dataset to a CSV file.
    """

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    data.to_csv(output_path / filename, index=False)


if __name__ == "__main__":
    print_synthetic_data_checks()
    save_synthetic_dataset()













# Step 8: 

# Step 9: Calculate the MSI-BPD total score.
# The total score is the sum of the 10 MSI-BPD items.

# Step 10: Classify BPD screening status.
# A score of 7 or higher indicates a positive BPD screening result.

# Step 11: Generate the WPI pain items.
# The WPI measures pain in 19 body areas during the past week.

# Step 12: Calculate the WPI total score.
# The WPI score is the sum of all painful body areas.

# Step 13: Generate the SSS symptom severity items.
# The SSS measures fatigue, unrefreshing sleep, cognitive symptoms, headaches, abdominal pain, and depression.

# Step 14: Calculate the SSS total score.
# The SSS score is the sum of all symptom severity items.

# Step 15: Generate symptom duration.
# Fibromyalgia symptoms must be present for at least three months.

# Step 16: Calculate generalized pain.
# Generalized pain is defined as pain in at least four out of five body regions.

# Step 17: Diagnose fibromyalgia based on ACR criteria.
# FM is diagnosed using the WPI score, SSS score, generalized pain, and symptom duration.

# Step 18: Increase FM probability in the BPD group.
# Participants with BPD are simulated as having a higher risk of FM.

# Step 19: Add gender as an additional predictor of FM.
# Women are simulated as having a slightly higher probability of FM.

# Step 20: Keep gender unrelated to BPD.
# Gender should predict FM but not BPD group membership.

# Step 21: Check descriptive statistics.
# Means, frequencies, and group differences are inspected before running the main analysis.

# Step 22: Test group matching.
# Age and gender are compared between the BPD and control groups.

# Step 23: Test the main association.
# A chi-square test is used to compare FM rates between groups.

# Step 24: Run logistic regression.
# FM diagnosis is predicted from BPD group, gender, and age.

# Step 25: Interpret the model.
# A significant BPD effect means that BPD predicts higher odds of FM.

# Step 26: Check the role of gender.
# A significant gender effect means that gender adds predictive value for FM.

# Step 27: Confirm that gender does not predict BPD.
# This supports the claim that gender is not confounded with group assignment.

# Step 28: Save the synthetic dataset.
# The final dataset is exported as a CSV file for analysis.
