import csv
from enum import Enum
import numpy as np

class Indices(Enum):
    id = 0
    name = 1
    first = 2
    last = 3
    sex = 5
    age = 7
    age_cat = 8
    race = 9
    priors_count = 14
    days_b_screening_arrest = 15
    c_charge_degree = 22
    is_recid = 24
    type_of_assessment = 38 # this is for General Recidivisim Risk
    decile_score = 39 # this is the decile score of General Recidivism Risk
    score_text = 40 # this is for General Recidivism Risk
    two_year_recid = 52 # same as is_recid

def read_csv(filename):
    with open(filename) as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        # next(reader, None)  # skip the headers
        data_read = [row for row in reader]
    return data_read

def preprocess(data):
    filtered = []
    for row in data[1:]:
        if row[Indices.days_b_screening_arrest.value]:
            # data quality: charge vs arrest date must be close enough
            if int(row[Indices.days_b_screening_arrest.value]) <= 30 and int(row[Indices.days_b_screening_arrest.value]) >= -30:
                # data quality: if the flag is -1, then there was no compas case
                if int(row[Indices.is_recid.value]) != -1:
                    # data quality: "O" degrees do not result in Jail time
                    if row[Indices.c_charge_degree.value] != "0":
                        # data quality: must have had 2 years outside jail or recidivated
                        if row[Indices.score_text.value] != "N/A":
                            # analysis: only considering those over 45 for now
                            if row[Indices.age_cat.value] == "Greater than 45":
                                # analysis: only considering men for now
                                if row[Indices.sex.value] == "Male":
                                    filtered.append(row)
    return filtered

def extract_prior_crimes(filtered):
    prior_crimes = []
    for row in filtered:
        # arrest bias only applies to black defendants
        if row[Indices.race.value] == "African-American":
            num_prior_crimes = int(row[Indices.priors_count.value])
            for i in range(0, num_prior_crimes):
                prior_crimes.append(row[Indices.id.value])
    return prior_crimes

def sample_crimes(original_prior_crimes, percent_arrest_bias):
    num_keep = (1 - percent_arrest_bias) * len(original_prior_crimes)
    return np.random.choice(original_prior_crimes, size=num_keep, replace=False)

def extract_recidivated(filtered):
    recidivated = []
    for row in filtered:
        if row[Indices.race.value] == "African-American":
            if row[Indices.two_year_recid.value] == "1":
                recidivated.append(row[Indices.id.value])
    return recidivated

# data[i] is info for the ith defendant; data[0] is the column headers
all_data = read_csv("compas-scores-two-years.csv")
filtered = preprocess(all_data)
original_prior_crimes = extract_prior_crimes(filtered)
original_recidivated = extract_recidivated(filtered)
# the below will go in a for loop 
sampled_prior_crimes = sample_crimes(original_prior_crimes, percent_arrest_bias = 0.03)
sampled_recidivated = sample_crimes(original_recidivated, percent_arrest_bias = 0.03)
