#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:42:15 2019

@author: Shaq9527
"""

import pandas as pd
import sys


arg = sys.argv
input_file = arg[1]

data = pd.read_csv(input_file)

all_cols = data.columns.values.tolist()

target_col = ['Value']


# =============================================================================
# drop some columns
# =============================================================================
drop_cols = ['Unnamed: 0', 'ID', 'Name', 'Photo', 'Flag', 'Overall', 
			 'Club Logo', 'Wage', 'Real Face',
			 'Jersey Number', 'Joined', 'Loaned From', 'Release CLause'
             ]

data = data.drop(drop_cols, axis=1)
data = data.dropna(axis=0, how='any')






encoding_cols = ['Nationality', 'Preferred Foot', 'Work Rate', 
				 'Body Type', 'Position',
				 ]


standardize_cols = ['Height', 'Weight']



discretize_cols = ['Age', 'Potential', 'Club', 'Special' 
                   ]




# standardize then bin
position_cols = ['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM',
              'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB',
			  'LB', 'LCB', 'CB', 'RCB', 'RB']


skill_cols = ['Crossing', 'Finishing', 'HeadingAccuracy',
			  'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 
			  'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 
			  'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 
			  'Strength', 'LongShots,Aggression', 'Interceptions', 'Positioning', 
			  'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 
			  'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 
			  'GKPositioning', 'GKReflexes']





data.to_csv("data_processed.csv", index=False)