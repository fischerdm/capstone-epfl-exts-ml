#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:29:26 2019

@author: davidfischer
"""

"""

import airbnb data and cleaning

"""

import pandas as pd
import numpy as np

def import_data(abs_path):
    return(pd.read_csv(abs_path, low_memory=False))
    
# Preprocessing the data
def prepro_dataset(data, prefix='House_', restrict_vars=None, restrict_obs=None, remove_na=False):
    
    """
    args:
        data: ``pandas dataframe``.
            The dataset.
        prefix: ``string``.
            Prefix to annotate the data.
        restrict_vars: ``list``.
            To restrict the variables.
        restrict_obs: ``tuple`` or ``None``, optional (default None).
            To restrict the dataset. (a, b) means that prices >= a and <= b are
            kept.
        remove_na: ``boolean``, optional (default False)
            If True, missing values are removed
        
    returns:
        the cleaned dataset
    
    """
    
    data = data.copy()
    
    # Get rid of variables and remove missing values
    if restrict_vars is not None:
        data = data.loc[:, restrict_vars]
    
    if remove_na:
        data.dropna(how='any', inplace=True)
    
    # lower case column names
    data.columns = data.columns.str.lower()
    
    # rename lattitude and longtitude
    try:
        data.rename(columns={'lattitude': 'latitude', 'longtitude': 'longitude'}, inplace=True)
    except:
        pass
    
    # clean suburbs
    try:
        data.loc[:, 'suburb'] = [el.lower().replace('.', '') for el in data.suburb]
    except:
        pass
    
    # exclude data
    if restrict_obs is not None:
        data = data.loc[(data.price >= restrict_obs[0]) & (data.price <= restrict_obs[1])]
        #data = data.loc[(data.price >= 10000) & (data.price <= 2500000), :]
    
    # dates
    try:
        data.loc[:, 'date'] = pd.to_datetime(data.loc[:, 'date'], format="%d/%m/%Y")
    except:
        pass
    
    # label the data
    colnames = list(data)
    colnames = [prefix + name for name in colnames]
    data.columns = colnames
    
    return(data)
    
   

def aggregate_data(data, values, by, agg_fun):
    
    """
     
    This function aggregates values indicates by `variables` by the different 
    house types and by some variables indicated by `by`. The house types are:
        
        h - house, cottage, villa, semi, terrace
        u - unit, duplex
        t - townhouse
    
    args:
        data: ``Pandas dataframe``.
            The dataset.
        values: ``list``.
            The name of the values that are aggregated.
        by: ``list``.
            The name(s) of the variable(s) that group(s) the data.
        agg_fun: ``string``.
            The function name to aggregate the data, e.g. 'mean', 'sum' ...
        
    returns:
        The aggregated dataframe.
    """
    
    data = data.loc[:, values + by + ['House_type']]
    
    data_agg = data.loc[:, values + by + ['House_type']].groupby(by=by + ['House_type']).agg(agg_fun)
    
    
    # Indexing
    data_agg.reset_index(inplace=True)
    data_agg.set_index(by, inplace=True)
    
    # Create the table
    data_agg = data_agg.pivot(columns='House_type', values=values)
    
    # The table has the form like this:
    
    #                   va1          var2
    # House_type    h   t   u      h  t  u
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
 
    
def aggregate_housing_market_data_circles(data, variables, obs, func):
    
    """
    args:
        data: ``DataFrame``
            The dataset.
        variables: ``list``
            The variables.
        obs: ``dictionary``
            The observations to aggregate.
        func: ``list``
            A list with aggregation functions as 'mean', 'median', 'count' etc.
    
    returns:
        a dictionary with the aggregation functions as key(s) and the aggregated 
        data as value(s).
        
    """
    
    data = data.copy()
    
    # to store the results
    results = dict()
    
    first = []
    for i in range(len(func)):
        first.append(True)

    for i,k in enumerate(obs.keys()):
    
        #values = ['House_price', 'House_rooms', 'House_distance', 'House_bedroom2',
        #          'House_bathroom', 'House_landsize']
        # now called variables
    
        try:
            tmp = data.loc[obs[k], variables + ['House_type']]
        except:
            pass
        
        #print(tmp)
    
        tmp['House_group'] = int(k)
        
        # Iteration over the different aggregation functions
        for j,f in enumerate(func):
            tmp_agg = aggregate_data(tmp, variables, ['House_group'], f)
            #tmp_agg_counts = house.aggregate_data(tmp, values, ['House_group'], 'count')
    
            if tmp_agg.shape[0] == 0:
                # empty obs at keys '106', '311', '406', '414' etc. for 5km radius.
                pass
    
            if first[j] == True and tmp_agg.shape[0] != 0:
                results[f] = tmp_agg.copy()
                first[j] = False
        
            elif first[j] == False and tmp_agg.shape[0] != 0:
                results[f] = pd.concat([results[f], tmp_agg], sort=True)
                #housing_agg_5km_counts = pd.concat([housing_agg_5km_counts, tmp_agg_counts], sort=True)
     
        #print(results)
        
    # rename columns
    colnames = []
    #for name in ['House_price', 'House_rooms', 'House_distance', 'House_bedroom2', 
    #         'House_bathroom', 'House_landsize']:
    for name in variables:
        for t in ['h', 't', 'u']:
            colnames.append(name + '_' + t)
    
    for k in results.keys():
        tmp = results[k].loc[:, ['House_group'] + colnames]
        tmp.columns = ['House_group'] + colnames
        tmp.index = range(tmp.shape[0])
        results[k] = tmp.copy()    
        
    return(results)   

# Source: https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas
def _haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def the_nearest_obs_housing(data1, data2, obs):
    
    """
    args:
        data: ``DataFrame``
            The latitudes, longitudes and of the first dataset.
        data2: ``DataFrame``
            The latitudes, longitudes and house type of the first dataset. 
            The house types are 'h', 'u' and 't'.
        obs: ``dictionary``.
            The observation candidates, correspondent to the second dataset.
            
    returns:
        A dictionary with the index of the nearest observation as value.
    """
    
    #data = data.copy()
    
    result_dict = {}
    
    
    for i in range(len(obs)):
        
        if len(obs[str(i)]) == 0:
            result_dict[str(i)] = {'h': None, 'u': None, 't': None}
            continue
        
        min_dist_1 = np.Inf # h
        min_dist_ob_1 = None
        min_dist_2 = np.Inf # u
        min_dist_ob_2 = None
        min_dist_3 = np.Inf # t
        min_dist_ob_3 = None
        
        for j in obs[str(i)]:
            
            try:
                room_type = data2.loc[j,'House_type'] # The room type of the observation of interest
                dist = _haversine_np(data1.loc[i,'House_longitude'], data1.loc[i,'House_latitude'], data2.loc[j,'House_longitude'], data2.loc[j,'House_latitude'])
            except:
                dist = np.Inf
            
            if room_type == 'h' and dist < min_dist_1 and j != i:
                min_dist_1 = dist
                min_dist_ob_1 = j
            elif room_type == 'u' and dist < min_dist_2 and j != i:
                min_dist_2 = dist
                min_dist_ob_2 = j
            elif room_type == 't' and dist < min_dist_3 and j != i:
                min_dist_3 = dist
                min_dist_ob_3 = j
                
        result_dict[str(i)] = {'h': min_dist_ob_1, 'u': min_dist_ob_2, 't': min_dist_ob_3}
    
    return(result_dict)
    

def the_nearest_obs_housing_prices(data1, data2, obs, var_names, distance=True):
    
    """
    args: 
        data1: ``DataFrame``
            The first dataset.
        data2: ``DataFrame``
            The second dataset.
        obs: ``dictionary of dictionaries``
            The observations. The keys correspond to the first and the
            values to the second dataset.
        var_names: ``list``
            The variable names. The names must be part of 'data'.
        distance: ``boolean``
            If True the distance is calculated.
        
    returns:
        A DataFrame with the airbnb prices for the observations and room types. 
        
    """
    
    data1 = data1.copy()
    data2 = data2.copy()
    
    data1['Air_group'] = range(data1.shape[0])
    data2['House_group'] = range(data2.shape[0])
    
    # NEW: dictionaries instead of lists
    idx_1_dict = dict()
    idx_2_dict = dict()
    idx_3_dict = dict()
    
    idx_1_values = []
    idx_2_values = []
    idx_3_values = []
    
    idx_1_keys = []
    idx_2_keys = []
    idx_3_keys = []
    
    for i in obs.keys():
        idx_1_dict[i] = obs[i]['h']
        idx_2_dict[i] = obs[i]['u']
        idx_3_dict[i] = obs[i]['t']
        
        idx_1_values.append(obs[i]['h'])
        idx_2_values.append(obs[i]['u'])
        idx_3_values.append(obs[i]['t'])
        
        idx_1_keys.append(int(i))
        idx_2_keys.append(int(i))
        idx_3_keys.append(int(i))
    
    df_key_value_1 = pd.DataFrame({'Air_group_data1': idx_1_keys,
                                  'House_group_data2_1': idx_1_values})
    df_key_value_2 = pd.DataFrame({'House_group_data2_2': idx_2_values})
    df_key_value_3 = pd.DataFrame({'House_group_data2_3': idx_3_values})
    
    df_key_value = pd.concat([df_key_value_1, df_key_value_2, df_key_value_3], axis=1)
    
    data_tmp_1 = data2.loc[data2.House_group.isin(idx_1_values), ['House_group'] + var_names]
    data_tmp_2 = data2.loc[data2.House_group.isin(idx_2_values), ['House_group'] + var_names]
    data_tmp_3 = data2.loc[data2.House_group.isin(idx_3_values), ['House_group'] + var_names]
    
    # Merges
    data_tmp_1.columns = [n + '_h' for n in ['House_group'] + var_names]
    data_tmp_2.columns = [n + '_u' for n in ['House_group'] + var_names]
    data_tmp_3.columns = [n + '_t' for n in ['House_group'] + var_names]
    
    tmp = df_key_value.merge(data_tmp_1, left_on='House_group_data2_1',
                            right_on='House_group_h', how='left')
    tmp.drop(columns='House_group_h', inplace=True)
    tmp = tmp.merge(data_tmp_2, left_on='House_group_data2_2',
                   right_on='House_group_u', how='left')
    tmp.drop(columns='House_group_u', inplace=True)
    tmp = tmp.merge(data_tmp_3, left_on='House_group_data2_3',
                   right_on='House_group_t', how='left')
    tmp.drop(columns='House_group_t', inplace=True)
    
    #print(tmp.head(5))
    
    if distance:
        
        dist_1 = []
        dist_2 = []
        dist_3 = []
        
        for g in data1.Air_group:
            
            if idx_1_dict[str(g)] is not None:
                dist_1.append(_haversine_np(data1.loc[g, 'Air_longitude'], 
                                                data1.loc[g, 'Air_latitude'],
                                                data2.loc[idx_1_dict[str(g)], 'House_longitude'], 
                                                data2.loc[idx_1_dict[str(g)], 'House_latitude']))
            else:
                dist_1.append(None)
            
            if idx_2_dict[str(g)] is not None:
                dist_2.append(_haversine_np(data1.loc[g, 'Air_longitude'], 
                                                data1.loc[g, 'Air_latitude'],
                                                data2.loc[idx_2_dict[str(g)], 'House_longitude'], 
                                                data2.loc[idx_2_dict[str(g)], 'House_latitude']))
            else:
                dist_2.append(None)
                
            if idx_3_dict[str(g)] is not None:
                dist_3.append(_haversine_np(data1.loc[g, 'Air_longitude'], 
                                            data1.loc[g, 'Air_latitude'],
                                            data2.loc[idx_3_dict[str(g)], 'House_longitude'], 
                                            data2.loc[idx_3_dict[str(g)], 'House_latitude']))
            else:
                dist_3.append(None)
        
        d_1 = {'dist': dist_1, 'Air_group': data1.Air_group}
        d_2 = {'dist': dist_2, 'Air_group': data1.Air_group}
        d_3 = {'dist': dist_3, 'Air_group': data1.Air_group}
        
        dist_1 = pd.DataFrame(data=d_1)
        dist_2 = pd.DataFrame(data=d_2)
        dist_3 = pd.DataFrame(data=d_3)
        
        # Merge distances
        data_tmp = tmp.merge(dist_1, left_on='Air_group_data1', right_on='Air_group',
                             how='left')
        data_tmp.rename(columns={'dist': 'dist_h'}, inplace=True)
        data_tmp = data_tmp.merge(dist_2, left_on='Air_group_data1', right_on='Air_group',
                             how='left')
        data_tmp.rename(columns={'dist': 'dist_u'}, inplace=True)
        data_tmp = data_tmp.merge(dist_3, left_on='Air_group_data1', right_on='Air_group',
                             how='left')
        data_tmp.rename(columns={'dist': 'dist_t'}, inplace=True)
    
    # sort by Air_group
    data_tmp.sort_values(by='Air_group', inplace=True)
    
    # cleaning
    data_tmp.drop(columns=['Air_group', 'Air_group_x', 'Air_group_y', 
                           'House_group_data2_1', 'House_group_data2_2',
                          'House_group_data2_3'], inplace=True)
    data_tmp.rename(columns={'Air_group_data1': 'Air_group'}, inplace=True)
    
    return data_tmp 