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

drop_cols = ['ID', 'Name', 'Photo', 'Flag', 'Club Logo', 'Loaned From', 
             ]


data = data.drop(drop_cols, axis=1)
data = data.dropna(axis=0, how='any')

discretize_cols = ['Age', 'Overall', 'Potential', 'Value', 'Wage', 'Special', 
                   ]

skill_cols = ['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'LCM', 'RAM',
              'LM', 'LCM',
                   ]

encoding_cols = [
                   ]