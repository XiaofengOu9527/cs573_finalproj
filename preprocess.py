#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:42:15 2019

@author: Shaq9527
"""

import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np


arg = sys.argv
input_file = arg[1]
flag = arg[2]

data = pd.read_csv(input_file)

all_cols = data.columns.values.tolist()

target_col = ['Value']


# =============================================================================
# drop some columns
# =============================================================================
drop_cols = ['Unnamed: 0', 'ID', 'Photo', 'Name', 'Flag', 'Overall', 
			 'Club Logo', 'Wage', 'Real Face', 'Jersey Number', 
			 'Joined', 'Loaned From', 'Release Clause', 'Contract Valid Until']

data = data.drop(drop_cols, axis=1)
data = data.dropna(axis=0, how='any').reset_index(drop=True)
data = data.sample(frac=1).reset_index(drop=True)

print("Number of datas after dropping empty value rows: {}".format(str(data.shape[0])))




encoding_cols = ['Nationality', 'Preferred Foot', 'Work Rate', 
				 'Body Type', 'Position', 'Club']

# alphabet encoding (for NBC)
if flag == 'nbc':
	attri_val = {}
	attri_numval = {}

	""" initialize """
	for attri in encoding_cols:
		attri_val[attri] = set()
		
		
	""" collect all attribute values """
	for index, row in data.iterrows():
		for attri in encoding_cols:
			attri_val[attri].add(row[attri])

	""" sort the values and generate a dict for each value """
	for attri in encoding_cols:
		attri_val[attri] = list(attri_val[attri])
		attri_val[attri].sort()
		attri_numval[attri] = dict(zip(attri_val[attri], range(len(attri_val[attri]))))
			
	""" convert the attribute to numerical value """
	for index, row in data.iterrows():
		for attri in encoding_cols:
			data.loc[index, attri]= int(attri_numval[attri][row[attri]])

# one hot encoding (for SVM, KNN, NN)
else:
	data = pd.get_dummies(data, columns=encoding_cols, drop_first=True)






standardize_cols = ['Value', 'Height', 'Weight']
discretize_cols = ['Age', 'Potential', 'Special', 'Height', 'Weight', 'Value']

positions = ['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 
			'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 
			'LB', 'LCB', 'CB', 'RCB', 'RB']

skills = ['Crossing', 'Finishing', 'HeadingAccuracy','ShortPassing', 
		 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 
		 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 
		 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 
		 'Strength', 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 
		 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 
		 'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 
		 'GKPositioning', 'GKReflexes']

VALUE = []
idxs_dropped = []
threshold = 1
for index, row in data.iterrows():

	# standardize ['Value', 'Height', 'Weight']
	value = row['Value'].split('€')[1]
	
	if len(value.split('M')[0]) == len(value) - 1:
		data.at[index, 'Value'] = round(float(value.split('M')[0]), 2)
	elif len(value.split('K')[0]) == len(value) - 1:
		data.at[index, 'Value'] = round(float(value.split('K')[0]) * 0.001, 2)
	else:	
		try:
			data.at[index, 'Value'] = float(value)
		except:
			raise ValueError('incorrect Value data format at {} row'.format(str(index)))

	if float(data.at[index, 'Value']) > threshold:
		VALUE.append(float(data.at[index, 'Value']))
	else:	
		idxs_dropped.append(index)
		continue

	height = row['Height']
	[feet, inch] = height.split("'")
	data.at[index, 'Height'] = float(feet) * 30.48 + float(inch) * 2.54
	
	weight = row['Weight']
	data.at[index, 'Weight'] = float(weight.split('lbs')[0])

	
	# standardize positions
	for position in positions:
		rating = list(row[position].split('+'))
		data.at[index, position] = float(rating[0])

data = data.drop(index=idxs_dropped).reset_index(drop=True)
print('Number of datas after dropping low value rows: {}'.format(data.shape[0]))

plt.hist(VALUE, bins=[0,2,4,6,8,10,15,20,30,40,60,80,120])
# plt.hist(VALUE, bins='fd')	
plt.ylabel('frequency')
plt.show()
	



bins = 5
labels = [1, 2, 3, 4, 5]

val_bins = [0,2,4,6,8,10,15,20,30,40,60,80,120]
val_labels = list(np.arange(len(val_bins) - 1) + 1)
# print(val_labels)
for col in discretize_cols:
	if col != 'Value':
		data[col] = pd.cut(data[col], bins=bins, labels=labels)
	else:
		data[col] = pd.cut(data[col], bins=val_bins, labels=val_labels)

for col in positions:
	data[col] = pd.cut(data[col], bins=[0, 20, 40, 60, 80, 100], labels=labels)

for col in skills:
	data[col] = pd.cut(data[col], bins=[0, 20, 40, 60, 80, 100], labels=labels)


if flag == 'nbc':
	data.to_csv("data_processed_nbc.csv", index=False)
else:
	data.to_csv("data_processed.csv", index=False)