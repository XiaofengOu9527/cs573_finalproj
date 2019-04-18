import pandas as pd
import numpy as np
import sys
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
import matplotlib.pyplot as plt


arg = sys.argv
input_file = arg[1]

full_data = pd.read_csv(input_file)
# full_data = full_data.sample(frac=1).reset_index(drop=True)

all_cols = full_data.columns.values.tolist()
target_col = ['Value']
other_cols = [col for col in all_cols if col not in target_col]

data = full_data[other_cols].to_numpy()
target = full_data[target_col].to_numpy().squeeze()



gnb = GaussianNB()
mnb = MultinomialNB(alpha=3.0)
cnb = ComplementNB(alpha=3.0)

test_frac = 0.2
test_full_data = full_data.sample(frac=test_frac)
train_full_data = full_data.drop(test_full_data.index)


test_data = test_full_data[other_cols].to_numpy()
test_target = test_full_data['Value'].to_numpy()



t_fracs = (0.4, 0.6, 0.8, 1.0)
random_seeds = np.random.randint(1000, size=10)



train_accuracy = np.zeros(shape=(3, len(t_fracs), len(random_seeds)))
test_accuracy = np.zeros(shape=(3, len(t_fracs), len(random_seeds)))

for i in range(len(t_fracs)):
      t_frac = t_fracs[i]
      for j in range(len(random_seeds)): 
            random_seed = random_seeds[j]

            sample_train_full_data = train_full_data.sample(frac=t_frac, random_state=random_seed)

            train_data = sample_train_full_data[other_cols].to_numpy()
            train_target = sample_train_full_data['Value'].to_numpy()

            #### Gaussian nbc
            gnb_model = gnb.fit(train_data, train_target)
            gnb_test_pred = gnb_model.predict(test_data)
            gnb_train_pred = gnb_model.predict(train_data)

            test_accuracy[0, i, j] = (gnb_test_pred == test_target).sum() / test_data.shape[0]

            train_accuracy[0, i, j] = (gnb_train_pred == train_target).sum() / train_data.shape[0]

            #### Multinomial nbc
            mnb_model = mnb.fit(train_data, train_target)
            mnb_test_pred = mnb_model.predict(test_data)
            mnb_train_pred = mnb_model.predict(train_data)

            test_accuracy[1, i, j] = (mnb_test_pred == test_target).sum() / test_data.shape[0]

            train_accuracy[1, i, j] = (mnb_train_pred == train_target).sum() / train_data.shape[0]

            #### Complement nbc
            cnb_model = cnb.fit(train_data, train_target)
            cnb_test_pred = cnb_model.predict(test_data)
            cnb_train_pred = cnb_model.predict(train_data)

            test_accuracy[2, i, j] = (cnb_test_pred == test_target).sum() / test_data.shape[0]

            train_accuracy[2, i, j] = (cnb_train_pred == train_target).sum() / train_data.shape[0]




train_aver_accuracy = np.mean(train_accuracy, axis=2)
train_std_error = np.std(train_accuracy, axis=2) / np.sqrt(len(random_seeds))

test_aver_accuracy = np.mean(test_accuracy, axis=2)
test_std_error = np.std(test_accuracy, axis=2) / np.sqrt(len(random_seeds))


plt.figure()
plt.errorbar(range(len(t_fracs)), train_aver_accuracy[0, :], yerr=train_std_error[0, :], label='train_gnb', linestyle='-')
plt.errorbar(range(len(t_fracs)), train_aver_accuracy[1, :], yerr=train_std_error[1, :], label='train_mnb', linestyle='-')
plt.errorbar(range(len(t_fracs)), train_aver_accuracy[2, :], yerr=train_std_error[2, :], label='train_cnb', linestyle='-')


plt.errorbar(range(len(t_fracs)), test_aver_accuracy[0, :], yerr=test_std_error[0, :], label='test_gnb', linestyle='-.')
plt.errorbar(range(len(t_fracs)), test_aver_accuracy[1, :], yerr=test_std_error[1, :], label='test_mnb', linestyle='-.')
plt.errorbar(range(len(t_fracs)), test_aver_accuracy[2, :], yerr=test_std_error[2, :], label='test_cnb', linestyle='-.')


plt.xticks(range(len(t_fracs)), t_fracs)
plt.legend()
plt.show()











# y_mnb_pred = mnb.fit(data, target).predict(data)
# print("MNB: number of correctly labeled points out of a total %d points : %d"
#       % (data.shape[0],(target == y_mnb_pred).sum()))
# print("MNB Accuracy: {}".format(str(round((target == y_mnb_pred).sum() / data.shape[0], 2))))



# y_gnb_pred = gnb.fit(data, target).predict(data)
# print("GNB: number of correctly labeled points out of a total %d points : %d"
#       % (data.shape[0],(target == y_gnb_pred).sum()))
# print("GNB Accuracy: {}".format(str(round((target == y_gnb_pred).sum() / data.shape[0], 2))))


# y_cnb_pred = cnb.fit(data, target).predict(data)
# print("CNB: number of correctly labeled points out of a total %d points : %d"
#       % (data.shape[0],(target == y_cnb_pred).sum()))
# print("CNB Accuracy: {}".format(str(round((target == y_cnb_pred).sum() / data.shape[0], 2))))