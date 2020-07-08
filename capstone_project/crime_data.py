#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:51:14 2019

@author: davidfischer
"""

"""

import crime data and cleaning

"""

import pandas as pd

def import_data(abs_path, sheet):

    return(pd.read_excel(abs_path, sheet_name=sheet))
    
# Clean data
def prepro_dataset(data):
    
    """
    args:
        data: ``pandas dataframe``.
            The dataset.
        
    returns:
        the cleaned dataset.
    
    """
    
    data = data.copy()
    
    # lower case column names. ' ' are replaced by _ for and ',' by ''
    s = [el.lower().replace(' ', '_').replace(',', '') for el in data.columns]
    data.columns = s
    
    # to lower case
    data.loc[:, 'local_government_area'] = data.loc[:, 'local_government_area'].str.lower()
    data.loc[:, 'police_service_area'] = data.loc[:, 'police_service_area'].str.lower()
    
    # label the data
    colnames = list(data)
    colnames = ['Crime_' + name for name in colnames]
    data.columns = colnames
    
    return(data)
    

def aggregate_data(data, values, by, crime_types, agg_fun):
    
    """
     
    This function aggregates values indicates by `variables` by the different 
    crime types and by some variables indicated by `by`. The house types are:
    
    args:
        data: ``Pandas dataframe``.
            The dataset.
        values: ``list``.
            The name of the values that are aggregated.
        by: ``list``.
            The name(s) of the variable(s) that group(s) the data.
        crime_types: ``string``.
            The name of the variable with the crime types.
        agg_fun: ``string``.
            The function name to aggregate the data, e.g. 'mean', 'sum' ...
        
    returns:
        The aggregated dataframe.
    """
    
    data = data.loc[:, values + by + [crime_types]]
    
    data_agg = data.loc[:, values + by + [crime_types]].groupby(by=by + [crime_types]).agg(agg_fun)
    
    
    # Indexing
    data_agg.reset_index(inplace=True)
    data_agg.set_index(by, inplace=True)
    
    # Create the table
    data_agg = data_agg.pivot(columns=crime_types, values=values)
    #data_agg_tmp = data_agg.copy()
    
    # The table has the form like this:
    
    #                     va1             var2
    # Crime_types    c1   c2   c3      c1  c2  c3
    # House_suburb
    # abbotsford
    # aberfeldie
    
    # 1. Get the levels of the 
    levels = data_agg.columns.levels[-1].values
    cl = [] # the columns
    for el in values:
        for l in levels:
            cl.append(el + '_' + l)
        
    data_agg.columns = cl
    
    # reset index
    data_agg.reset_index(inplace=True)
    
    # Replace NaN by 0
    #data_agg.fillna(0, inplace=True)
    
    return(data_agg)

