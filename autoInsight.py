#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:53:27 2019

@author: Thada.Ji
"""


import pandas as pd
import numpy as np
import xgboost
import re
import warnings
warnings.filterwarnings('ignore')

def binary_auto_insight(dataset, labelCol, positive_class):
    # train XGBoost model
    ntrees = 5
    dataset.columns = dataset.columns.str.replace(r"[^a-zA-Z0-9]", '_', regex = True)
    dataset[labelCol]  = np.where(dataset[labelCol] == positive_class, 1 ,0 )
    data = dataset.drop(columns=[labelCol])
    labels =  dataset[labelCol] 
    onePercent = int(np.round(data.shape[0]/100))
    params = {'lambda': 0, 'learning_rate': 1, 'max_depth': 1, 'min_child_weight': onePercent, 
              'objective': 'binary:logistic', 'seed': 9}
    model = xgboost.train(params, xgboost.DMatrix(data, label = labels), num_boost_round = ntrees)
    
    # Extract infomation of each tree
    j = model.get_dump()
    dataGrouped = dataset.copy()
    groups = model.predict(xgboost.DMatrix(data, label = labels), pred_leaf = True)
    allTrees = []
    for i in range(ntrees):
        text = j[i].splitlines()[0]
        splitText = re.split(',| ',text)
        colName = re.search(r"\[.*\]", splitText[0]).group()
        leaveStates = [s.split('=') for s in splitText[1:]]
        colDict = {1: ", ".join([s[0] for s in leaveStates if s[1]=='1']), 
                   2:  ", ".join([s[0] for s in leaveStates if s[1]=='2']) }
        dataGrouped[colName] = groups[:,i] 
        dataGrouped[colName] = dataGrouped[colName].apply(lambda x: colDict.get(x))
        allTrees.append(colName)
        
    # Display insights    
    label_col_pct = '{}(%)'.format(labelCol)
    format_dict = {label_col_pct: '{:.2%}'}
    from IPython.display import display, HTML
    for i in range(ntrees):
        riskTable = dataGrouped.groupby(allTrees[:(i+1)])[labelCol].agg(['sum','count'])
        riskTable['prob'] = 1.0*riskTable['sum']/riskTable['count']

        riskTable = riskTable.rename(columns={'sum':labelCol,
                              'count':'Total',
                              'prob':label_col_pct})
            
        #.set_table_styles([{'selector': 'th', 'props': [('font-size', '12pt'),('border-style','solid'),('border-width','1px')]}])
        display(riskTable.style.format(format_dict)
                .background_gradient(axis = 0, subset = pd.IndexSlice[:, ['{}(%)'.format(labelCol)]])
                .set_table_styles([{'selector': 'th', 'props': [('border-style','dotted'),('border-width','0.1px')]}])
                )
