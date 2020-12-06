import csv
from enum import Enum
import numpy as np
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass
import matplotlib.pyplot as plt
from collections import Counter

PERCENT_ARREST_BIAS  = 0.50

@dataclass
class ConfusionMatrix:
    """Class for keeping track of fp,fn,tp,tn."""
    tp: float
    tn: float
    fp: float
    fn: float

    def get_accuracy(self) -> float:
        return (self.tp + self.tn)/(self.tp + self.tn + self.fp + self.fn)

    def get_fpr(self) -> float:
        return self.fp/(self.fp + self.tn)

    def get_fnr(self) -> float:
        return self.fn/(self.fn + self.tp)

    def print_stats(self) -> float:
        print("Accuracy:", self.get_accuracy())
        print("False positive rate:", self.get_fpr())
        print("False negative rate:", self.get_fnr())
        print("Number recidivated:", self.tp + self.fn)
        print("Number not recidivated:", self.fp + self.tn)
        print("Number given medium/high risk score:", self.tp + self.fp)
        print("Number given low risk score:", self.tn + self.fn)

    def combine_matrices(self, other) -> float:
        return ConfusionMatrix(self.tp + other.tp, self.tn + other.tn, self.fp + other.fp, self.fn + other.fn)


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
                            # analysis: only considering a specific age range
                            age = int(row[Indices.age.value])
                            if age > 20 and age < 60:
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
    num_keep = int((1 - percent_arrest_bias) * len(original_prior_crimes))
    return np.random.choice(original_prior_crimes, size=num_keep, replace=False)

def extract_recidivated(filtered):
    recidivated = []
    for row in filtered:
        if row[Indices.race.value] == "African-American":
            if row[Indices.two_year_recid.value] == "1":
                recidivated.append(row[Indices.id.value])
    return recidivated

def prior_to_score_regression(filtered):
    num_priors = []
    compas_risk_scores = []
    for row in filtered:
        num_priors.append(int(row[Indices.priors_count.value]))
        compas_risk_scores.append(int(row[Indices.decile_score.value]))
    num_priors = np.array(num_priors).reshape(-1, 1)
    compas_risk_scores = np.array(compas_risk_scores)
    reg = LinearRegression().fit(num_priors, compas_risk_scores)
    print("R^2 of linear regression model:", reg.score(num_priors, compas_risk_scores))
    plt.scatter(num_priors, compas_risk_scores)
    plt.show()
    print(reg.intercept_)
    print(reg.coef_)
    return reg

# the statistics for white defendants do not change
def calculate_white_statistics(reg, filtered):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for row in filtered:
        if row[Indices.race.value] == "Caucasian":
            num_priors = int(row[Indices.priors_count.value])
            proj_risk_score = reg.predict(np.array([num_priors]).reshape(-1,1))[0]
            # recidivated
            if row[Indices.is_recid.value] == "1":
                if proj_risk_score < 5:
                    fn += 1
                else:
                    tp += 1
            # did not recidivate
            else:
                if proj_risk_score < 5:
                    tn += 1
                else:
                    fp += 1
    return ConfusionMatrix(tp, tn, fp, fn)

# the statistics for black defendants change due to arrest rate bias
def calculate_black_statistics(reg, prior_crimes, recidivated, filtered):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for row in filtered:
        if row[Indices.race.value] == "African-American":
            id = row[Indices.id.value]
            num_priors = np.count_nonzero(prior_crimes == id)
            proj_risk_score = reg.predict(np.array([num_priors]).reshape(-1,1))[0]
            # recidivated
            if id in set(recidivated):
                if proj_risk_score < 5:
                    fn += 1
                else:
                    tp += 1
            # did not recidivate
            else:
                if proj_risk_score < 5:
                    tn += 1
                else:
                    fp += 1
    return ConfusionMatrix(tp, tn, fp, fn)


# data[i] is info for the ith defendant; data[0] is the column headers
all_data = read_csv("compas-scores-two-years.csv")
ages = []
for row in all_data:
    ages.append(row[Indices.age.value])
occurrence_count = Counter(ages)
print(occurrence_count.most_common(1)[0][0])
filtered = preprocess(all_data)
print("Number of defendants:", len(filtered))
black_original_prior_crimes = np.array(extract_prior_crimes(filtered))
black_original_recidivated = np.array(extract_recidivated(filtered))
reg = prior_to_score_regression(filtered)
white_cm = calculate_white_statistics(reg, filtered)
print("White Defendant Statistics----------")
white_cm.print_stats()
print("Original Black Defendant Statistics----------")
black_cm_orig = calculate_black_statistics(reg, black_original_prior_crimes, black_original_recidivated, filtered)
black_cm_orig.print_stats()


# the below will go in a for loop
sampled_prior_crimes = sample_crimes(black_original_prior_crimes, percent_arrest_bias = PERCENT_ARREST_BIAS)
sampled_recidivated = sample_crimes(black_original_recidivated, percent_arrest_bias = PERCENT_ARREST_BIAS)
black_cm = calculate_black_statistics(reg, sampled_prior_crimes, sampled_recidivated, filtered)
print("Black Defendant Statistics---------")
black_cm.print_stats()
total_cm = white_cm.combine_matrices(black_cm)
print("Combined Statistics--------")
total_cm.print_stats()
