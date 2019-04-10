import pandas as pd
import numpy as np
import sys
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB


arg = sys.argv
input_file = arg[1]

full_data = pd.read_csv(input_file)
full_data = full_data.sample(frac=1).reset_index(drop=True)

all_cols = full_data.columns.values.tolist()
target_col = ['Value']
other_cols = [col for col in all_cols if col not in target_col]

data = full_data[other_cols].to_numpy()
target = full_data[target_col].to_numpy()

print(data[123:132,:])

gnb = GaussianNB()
cnb = ComplementNB()
mnb = MultinomialNB()

# y_gnb_pred = gnb.fit(data, target).predict(data)
# print("GNB: number of mislabeled points out of a total %d points : %d"
#       % (data.shape[0],(target != y_gnb_pred).sum()))

y_cnb_pred = cnb.fit(data, target).predict(data)
print("CNB: number of mislabeled points out of a total %d points : %d"
      % (data.shape[0],(target != y_cnb_pred).sum()))

y_mnb_pred = gnb.fit(data, target).predict(data)
print("MNB: number of mislabeled points out of a total %d points : %d"
      % (data.shape[0],(target != y_mnb_pred).sum()))