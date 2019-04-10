import pandas as pd
import numpy as np
import sys
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB


arg = sys.argv
input_file = arg[1]

full_data = pd.read_csv(input_file)
# full_data = full_data.sample(frac=1).reset_index(drop=True)

all_cols = full_data.columns.values.tolist()
target_col = ['Value']
other_cols = [col for col in all_cols if col not in target_col]

data = full_data[other_cols].to_numpy()
target = full_data[target_col].to_numpy().squeeze()

print([target.shape, data.shape])



gnb = GaussianNB()
cnb = ComplementNB()
mnb = MultinomialNB()

# y_gnb_pred = gnb.fit(data, target).predict(data)
# print("GNB: number of  correctly labeled points out of a total %d points : %d"
#       % (data.shape[0],(target == y_gnb_pred).sum()))

# y_cnb_pred = cnb.fit(data, target).predict(data)
# print("CNB: number of mislabeled points out of a total %d points : %d"
#       % (data.shape[0],(target == y_cnb_pred).sum()))

y_mnb_pred = mnb.fit(data, target).predict(data)
print("MNB: number of correctly labeled points out of a total %d points : %d"
      % (data.shape[0],(target == y_mnb_pred).sum()))
print("MNB Accuracy: {}".format(str(round((target == y_mnb_pred).sum() / data.shape[0], 2))))



y_gnb_pred = gnb.fit(data, target).predict(data)
print("GNB: number of correctly labeled points out of a total %d points : %d"
      % (data.shape[0],(target == y_gnb_pred).sum()))
print("GNB Accuracy: {}".format(str(round((target == y_gnb_pred).sum() / data.shape[0], 2))))


y_cnb_pred = cnb.fit(data, target).predict(data)
print("CNB: number of correctly labeled points out of a total %d points : %d"
      % (data.shape[0],(target == y_cnb_pred).sum()))
print("CNB Accuracy: {}".format(str(round((target == y_cnb_pred).sum() / data.shape[0], 2))))