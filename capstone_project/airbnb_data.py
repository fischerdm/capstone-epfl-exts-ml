#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 09:29:30 2019

@author: davidfischer
"""

"""

import airbnb data and cleaning

"""

import pandas as pd
import numpy as np

def import_data(abs_path):
    return(pd.read_csv(abs_path, low_memory=False))
    
# Clean data
def prepro_dataset(data, prefix='Air_', restrict=None):
    
    """
    args:
        data: ``pandas dataframe``.
            The dataset.
        prefix: ``string``.
            To annotate the data.
        restrict: tuple or None (default).
            To restrict the dataset. (a, b) means that prices >= a and <= b are
            kept.
        
    returns:
        the cleaned dataset.
    
    """
    
    # numeric data or dates:
    ###########################################################################
    
    # last_scraped, host_since, host_response_rate, host_listings_count, 
    # host_total_listings_count, latitude, longitude, accommodates, bathrooms, 
    # bedrooms, beds, square_feet, price, weekly_price, monthly_price, 
    # security_deposit, cleaning_fee, guests_included, extra_people, 
    # minimum_nights, maximum_nights, availability_30, availability_60, 
    # availability_90, availability_365, calendar_last_scraped, 
    # number_of_reviews, first_review, last_review, review_scores_rating, 
    # review_scores_accuracy, review_scores_cleanliness, review_scores_checkin, 
    # review_scores_communication, review_scores_location, review_scores_value, 
    # calculated_host_listings_count, reviews_per_month
    
    
    data=data.copy()
    
    # prices
    # -------------------------------------------------------------------------
    for name in ('price', 'weekly_price', 'monthly_price', 'security_deposit',
                 'cleaning_fee', 'extra_people'):
        data.loc[:, name] = data.loc[:, name].str.replace('$', '')
        data.loc[:, name] = data.loc[:, name].str.replace(',', '').astype(float)
    
    # dates
    # -------------------------------------------------------------------------
    for name in ('last_scraped', 'calendar_last_scraped', 'first_review', 
                 'last_review'):
        data.loc[:, name] = pd.to_datetime(data.loc[:, name], format="%Y-%m-%d")
    
    
    # host_response_rate
    # -------------------------------------------------------------------------
    data.loc[:, 'host_response_rate'] = data.host_response_rate.str.replace('%', '')
    data.loc[:, 'host_response_rate'] = data.host_response_rate.astype(float)
    
    # other numeric variables
    # -------------------------------------------------------------------------
    for name in ('host_listings_count', 'host_total_listings_count',
                 'latitude', 'longitude', 'accommodates', 'bathrooms', 'bedrooms',
                 'beds', 'square_feet', 'guests_included', 'minimum_nights', 
                 'maximum_nights', 'availability_30', 'availability_60',
                 'availability_90', 'availability_365', 'number_of_reviews',
                 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
                 'review_scores_checkin', 'review_scores_communication', 
                 'review_scores_location', 'review_scores_value', 'calculated_host_listings_count',
                 'reviews_per_month'):
        data.loc[:, name] = data.loc[:, name].astype(float)
    
    # Location
    ###########################################################################
    
    # Rk: Smart location is used as the variable that defines the location
    
    # smart location 
    data['smart_location_cleaned'] = data.smart_location
    data.loc[:, 'smart_location_cleaned'] = [el.split(',')[0].strip().lower().replace('.', '') for el in data.smart_location_cleaned]
    
    smart_locs_repl = {'smart_location_cleaned':
                   {'albert park melbourne': 'albert park',
                    'brunswick / melbourne': 'brunswick',
                    'brunswick vic 3056': 'brunkswick',
                    'chum creek': 'healsville',
                    'chum creek/healesville': 'healsville', # => chum creek not in 'housing_market'
                    'city of port phillip': 'port city phillip',
                    'coburg (melbourne)': 'coburg',
                    'doncaster vic 3108': 'doncaster',
                    'doncaster，melbourne': 'doncaster',
                    'east brunswick': 'brunswick east',
                    'east doncaster': 'doncaster east',
                    'east st kilda': 'st kilda east',
                    'ivanhoe (melbourne)': 'ivanhoe',
                    'melborne': 'melbourne',
                    'melbourne (eltham)': 'melbourne',
                    'melbourne cbd': 'melbourne',
                    'melbourne city': 'melbourne',
                    'melbourne vic 3000': 'melbourne',
                    'melbourne vic 3004': 'melbourne',
                    'melbourne victoria': 'melbourne',
                    'melton south ( strathtulloh)': 'melton south',
                    'middle park melbourne': 'middle park',
                    'mt dandenong': 'mount dandegong',
                    'mt waverley': 'mount waverly',
                    'prahran / toorak': 'prahran',
                    'ripponlea (east st kilda)': 'st kilda', # => ripponela not in 'housing_market'
                    'saint albans': 'st albans',
                    'saint andrews': 'st andrews',
                    'saint helena': 'st helena',
                    'saint kilda': 'st kilda',
                    'saint kilda beach': 'st kilda beach',
                    'saint kilda east': 'st kilda east',
                    'saint kilda west': 'st kilda west',
                    'somerton vic 3062': 'somerton',
                    'south yarra vic 3141': 'south yarra',
                    'southbank melbourne': 'southbank',
                    'st kilda / elwood': 'st kilda',
                    'st kilda west melbourne': 'st kilda',
                    'stkilda east': 'st kilda east',
                    'strathtulloh': 'melton south',
                    'strthtulloch': 'melton south',
                    'wantirna south vic 3152': 'wantirna south',
                    'west melbourne - flagstaff': 'west melbourne',
                    'wheelers hill vic 3150': 'wheelers hill'
                   }
                }
    
    #data = data.replace(smart_locs_repl)
    data.replace(smart_locs_repl, inplace=True)
    
    # exclude data
    if restrict is not None:
        #data = data.loc[(data.price >= 20) & (data.price <= 300), :]
        data = data.loc[(data.price >= restrict[0]) & (data.price <= restrict[1])]
    
    # label the data
    colnames = list(data)
    colnames = [prefix + name for name in colnames]
    data.columns = colnames
    
    return(data)
    
    
def _circle_candidates(data_1, data_2, max_distance):
    """
    args:
        data_1: numpy array.
            Stores the latitudes and longitudes of the first dataset.
        data_2: numpy array.
            Stores the latitudes and longitudes of the second dataset.
        max_distance: float.
            Defines the maximum distance between the location in data_1 and data_2.
            
    returns:
        A dictionary that stores for every observation i of data_1 the candidates 
        of data_2, j=1, ..., n, that potentially lie within a radius of max_distance 
        from the observation i.
    
    """
    
    # Shapes
    # ------
    
    #data_1_len = data_1.shape[0]
    #data_2_len = data_2.shape[0]
    
    # MATH
    # ==================================
    
    # Distances
    # ---------
    #map(np.radians, [longitude1, latitude1, longitude2, latitude2])

    #dlon = longitude1 - longitude2
    #dlat = latitude1 - latitude2

    #a = np.sin(dlat/2.0)**2 + np.cos(latitude1) * np.cos(latitude2) * np.sin(dlon/2.0)**2

    #c = 2 * np.arcsin(np.sqrt(a))
    #km = 6367 * c
    
    
    # How to get from km to dlon and dlat, when dlat or dlon, respectively are 0?
    # ---------------------------------------------------------------------------
    
    #c = 2 * np.arcsin(np.sqrt(a))
    #km = 6367 * c
    
    # => a = ?
    # km = 6367 * 2 * np.arcsin(np.sqrt(a))
    # km / 6367 / 2 = np.arcsin(np.sqrt(a))
    # [sin(km / 6367 / 2)]^2 = a
    
    # dlon = 0
    # dlat = 2*np.arcsin(np.sqrt(a))
    
    # dlat = 0
    # dlon = 2*np.arcsin(np.sqrt(np.arcsin(a / (np.cos(lat))**2)))
    
    lat1 = data_1[:,0]
    lon1 = data_1[:,1]
    lat2 = data_2[:,0]
    lon2 = data_2[:,1]
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    a = (np.sin(max_distance / (6367 * 2)))**2
    dlat = 2*np.arcsin(np.sqrt(a))
    #dlat = np.tile(dlat, data_1.shape[0])
    dlon = 2*np.arcsin(np.sqrt(np.arcsin(a/(np.cos(lat1)**2))))
    
    
    index_candidates_dict = {}
    obs = np.arange(0, data_2.shape[0])
    for i in range(data_1.shape[0]):
        index_candidates_dict[str(i)] = obs[(np.abs(lat1[i] - lat2) <= dlat) & \
                                            (np.abs(lon1[i] - lon2) <= dlon[i])]
    
    
    return(index_candidates_dict)
    
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


def circle_observations(data_1, data_2, max_distance):
    """
    This function deletes candidates that do not fulfill the max distance condition.
    
    args:
        data_1: ``numpy array``
            The latitudes and longitudes of the first dataset.
        data_2: ``numpy array``
            The latitudes and longitudes of the second dataset.
        max_distance: float.
            The max distance.
            
    returns:
        The cleaned dictionary.
    
    """
    
    # Create the circle candidates
    candidates_dict = _circle_candidates(data_1, data_2, max_distance)
    
    
    for i in range(len(candidates_dict)):
    #for i in range(5):
        #print(i)
        if len(candidates_dict[str(i)]) == 0: # was len(candiates_dict) == 0
            continue
        
        list_tmp = []
        for j in candidates_dict[str(i)]:
            #print(haversine_np(data_1[i,1], data_1[i,0], data_2[j,1], data_2[j, 0]))
            if _haversine_np(data_1[i,1], data_1[i,0], data_2[j,1], data_2[j,0]) <= max_distance:
                list_tmp.append(j)
        #print(list_tmp)   
        candidates_dict[str(i)] = np.array(list_tmp)
    
    return(candidates_dict)

# Old version
#def aggregate_data(data, values, by, agg_fun):
#    
#    """
#     
#    This function aggregates values indicates by `variables` by the different 
#    room types and by some variables indicated by `by`. The room types are:
#
#        - entire home/apt
#        - private room
#        - shared room
#    
#    args:
#        data: ``Pandas dataframe``.
#            The dataset.
#        values: ``list``.
#            The name of the values that are aggregated.
#        by: ``list``.
#            The name(s) of the variable(s) that group(s) the data.
#        agg_fun: ``string``.
#            The function name to aggregate the data, e.g. 'mean', 'sum' ...
#        
#    returns:
#        The aggregated dataframe.
#    """
#    
#    data = data.loc[:, values + by + ['Air_room_type']]
#    
#    data_agg = data.loc[:, values + by + ['Air_room_type']].groupby(by=by + ['Air_room_type']).agg(agg_fun)
#    
#    
#    # Indexing
#    data_agg.reset_index(inplace=True)
#    data_agg.set_index(by, inplace=True)
#    
#    # Create the table
#    data_agg = data_agg.pivot(columns='Air_room_type', values=values)
#    
#    
#    # 1. Get the levels of the 
#    levels = data_agg.columns.levels[-1].values
#    cl = [] # the columns
#    for el in values:
#        for l in levels:
#            cl.append(el + '_' + l)
#        
#    data_agg.columns = cl
#    
#    # reset index
#    data_agg.reset_index(inplace=True)
#    
#    # Replace NaN by 0
#    #data_agg.fillna(0, inplace=True)
#    
#    return(data_agg)
    
def aggregate_data(data, values, by, agg_fun, classifier):
    
    """
     
    This function aggregates values by some variables indicated by `by`. `Classifier`
    further groups the results on the columns. Choices for `classifier` are e.g.
    `Air_room_type` or `Air_property_type_2`. 
    
    args:
        data: ``Pandas dataframe``
            The dataset.
        values: ``list``
            The name of the values that are aggregated.
        by: ``list``
            The name(s) of the variable(s) that group(s) the data.
        agg_fun: ``string``
            The function name to aggregate the data, e.g. 'mean', 'sum' ...
        classifier: ``string``
            The variable that is used to further goup the values on the columns.
        
    returns:
        The aggregated dataframe.
    """
    
    data = data.loc[:, values + by + [classifier]]
    
    data_agg = data.loc[:, values + by + [classifier]].groupby(by=by + [classifier]).agg(agg_fun)
    
    
    # Indexing
    data_agg.reset_index(inplace=True)
    data_agg.set_index(by, inplace=True)
    
    # Create the table
    data_agg = data_agg.pivot(columns=classifier, values=values)
    
    
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

    
def aggregate_airbnb_data_circles(data, variables, obs, func, classifier):
    
    """
    args:
        data: ``DataFrame``
            The dataset.
        variables: ``list``
            The variables to aggregate.
        obs: ``dictionary``
            The observations to aggregate including the center point.
        func: ``list``
            A list with aggregation functions as 'mean', 'median', 'count' etc.
        classifier: ``string``
            The variable that is used to further group the values (columns).
        #categories: ``list``
        #    The categories of ``classifier``.
            
    
    returns:
        a dictionary with the aggregation functions as key(s) and the aggregated 
        data as value(s).
        
    """
    
    # Remove center from 'obs'
    obs_wo_center = obs.copy()

    for k in obs.keys():
        obs_wo_center[k] = np.delete(obs[k], np.where(obs[k] == int(k)))
    
    data = data.copy()
    
    # to store the results
    results = dict()
    
    first = []
    for i in range(len(func)):
        first.append(True)

    for i,k in enumerate(obs.keys()):
    
        values = variables 
    
        try:
            tmp = data.loc[obs_wo_center[k], values + [classifier]]
        except:
            pass
        
        #print(tmp)
    
        tmp['Air_group'] = int(k)
        
        # Iteration over the different aggregation functions
        for j,f in enumerate(func):
            tmp_agg = aggregate_data(tmp, values, ['Air_group'], f, classifier)
    
            #if tmp_agg.shape[0] == 0:
            #    pass
    
            if first[j] == True and tmp_agg.shape[0] != 0:
                results[f] = tmp_agg.copy()
                first[j] = False
        
            elif first[j] == False and tmp_agg.shape[0] != 0:
                results[f] = pd.concat([results[f], tmp_agg], sort=True)
        
    # rename columns
    for k in results.keys():
        colnames = results[k].columns.to_list()
        colnames.remove('Air_group')
        tmp = results[k].loc[:, ['Air_group'] + colnames]
        #print(tmp)
        tmp.columns = ['Air_group'] + colnames
        tmp.index = range(tmp.shape[0])
        results[k] = tmp.copy()    
        
    return(results)     


def the_nearest_obs(data1, data2, obs, classifier):
    
    """
    args:
        data: ``DataFrame``
            The latitudes, longitudes of the first dataset.
        data2: ``DataFrame``
            The latitudes, longitudes and room or property type of the first dataset. 
            The room types are 'Entire home/apt', 'Private room', 
            and 'Shared room', and the property types are 'Other', 'House_Cottage_Villa',
            'Apartment_Condominium', 'Townhouse'.
        obs: ``dictionary``.
            The observation candidates, correspondent to the second dataset.
        classifier: ``string``
            The type. Either 'Air_room_type' or 'Air_property_type_2'.
            
    returns:
        A dictionary with the index of the nearest observation as value.
    """
    
    data1 = data1.copy()
    data2 = data2.copy()
        
    result_dict = {}
    
    if classifier == 'Air_room_type':
    
        for i in range(len(obs)):
        
            if len(obs[str(i)]) == 0:
                result_dict[str(i)] = {'Entire home/apt': None, 'Private room': None, 'Shared room': None}
                continue
        
            min_dist_1 = np.Inf # Entire home/apt
            min_dist_ob_1 = None
            min_dist_2 = np.Inf # Private room
            min_dist_ob_2 = None
            min_dist_3 = np.Inf # Shared room
            min_dist_ob_3 = None
        
            for j in obs[str(i)]:
            
                room_type = data2.loc[j,'Air_room_type'] # The room type of the observation of interest
                try:
                    dist = _haversine_np(data1.loc[i,'Air_longitude'], data1.loc[i,'Air_latitude'], data2.loc[j,'Air_longitude'], data2.loc[j,'Air_latitude'])
                except:
                    dist = np.Inf
            
                if room_type == 'Entire home/apt' and dist < min_dist_1 and data1.loc[i, 'Air_id'] != data2.loc[j, 'Air_id']:
                    min_dist_1 = dist
                    min_dist_ob_1 = j
                elif room_type == 'Private room' and dist < min_dist_2 and data1.loc[i, 'Air_id'] != data2.loc[j, 'Air_id']:
                    min_dist_2 = dist
                    min_dist_ob_2 = j
                elif room_type == 'Shared room' and dist < min_dist_3 and data1.loc[i, 'Air_id'] != data2.loc[j, 'Air_id']:
                    min_dist_3 = dist
                    min_dist_ob_3 = j
                
            result_dict[str(i)] = {'Entire home/apt': min_dist_ob_1, 'Private room': min_dist_ob_2, 'Shared room': min_dist_ob_3}
    
    elif classifier == 'Air_property_type_2':
    
        for i in range(len(obs)):
        
            if len(obs[str(i)]) == 0:
                result_dict[str(i)] = {'Other': None, 'House_Cottage_Villa': None, 
                                       'Apartment_Condominium': None, 'Townhouse': None}
                continue
        
            min_dist_1 = np.Inf # Other
            min_dist_ob_1 = None
            min_dist_2 = np.Inf # House_Cottage_Villa
            min_dist_ob_2 = None
            min_dist_3 = np.Inf # Apartment_Condominium
            min_dist_ob_3 = None
            min_dist_4 = np.Inf # Townhouse
            min_dist_ob_4 = None
        
            for j in obs[str(i)]:
            
                property_type = data2.loc[j,'Air_property_type_2'] # The room type of the observation of interest
                try:
                    dist = _haversine_np(data1.loc[i,'Air_longitude'], data1.loc[i,'Air_latitude'], data2.loc[j,'Air_longitude'], data2.loc[j,'Air_latitude'])
                except:
                    dist = np.Inf
            
                if property_type == 'Other' and dist < min_dist_1 and data1.loc[i, 'Air_id'] != data2.loc[j, 'Air_id']:
                    min_dist_1 = dist
                    min_dist_ob_1 = j
                elif property_type == 'House_Cottage_Villa' and dist < min_dist_2 and data1.loc[i, 'Air_id'] != data2.loc[j, 'Air_id']:
                    min_dist_2 = dist
                    min_dist_ob_2 = j
                elif property_type == 'Apartment_Condominium' and dist < min_dist_3 and data1.loc[i, 'Air_id'] != data2.loc[j, 'Air_id']:
                    min_dist_3 = dist
                    min_dist_ob_3 = j
                elif property_type == 'Townhouse' and dist < min_dist_4 and data1.loc[i, 'Air_id'] != data2.loc[j, 'Air_id']:
                    min_dist_4 = dist
                    min_dist_ob_4 = j 
                
            result_dict[str(i)] = {'Other': min_dist_ob_1, 
                                   'House_Cottage_Villa': min_dist_ob_2, 
                                   'Apartment_Condominium': min_dist_ob_3,
                                   'Townhouse': min_dist_ob_4}
    
    
    return(result_dict) 
    

def the_nearest_obs_prices(data1, data2, obs, var_names,
                            classifier, distance=True):
    
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
        classifier: ``string``
            The type. Either 'Air_room_type' or 'Air_property_type_2'.
            
        
    returns:
        A DataFrame with the airbnb prices for the observations and room types. 
        
    """
    
    data1 = data1.copy()
    data2 = data2.copy()
    
    data1['Air_group'] = range(data1.shape[0])
    data2['Air_group'] = range(data2.shape[0])
    
    # ----------------------
    # Room type
    if classifier == "Air_room_type":
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
            idx_1_dict[i] = obs[i]['Entire home/apt']
            idx_2_dict[i] = obs[i]['Private room']
            idx_3_dict[i] = obs[i]['Shared room']
        
            idx_1_values.append(obs[i]['Entire home/apt'])
            idx_2_values.append(obs[i]['Private room'])
            idx_3_values.append(obs[i]['Shared room'])
        
            idx_1_keys.append(int(i))
            idx_2_keys.append(int(i))
            idx_3_keys.append(int(i))
    
        df_key_value_1 = pd.DataFrame({'Air_group_data1': idx_1_keys,
                                      'Air_group_data2_1': idx_1_values})
        df_key_value_2 = pd.DataFrame({'Air_group_data2_2': idx_2_values})
        df_key_value_3 = pd.DataFrame({'Air_group_data2_3': idx_3_values})
    
        df_key_value = pd.concat([df_key_value_1, df_key_value_2, df_key_value_3], axis=1)
    
        data_tmp_1 = data2.loc[data2.Air_group.isin(idx_1_values), ['Air_group'] + var_names]
        data_tmp_2 = data2.loc[data2.Air_group.isin(idx_2_values), ['Air_group'] + var_names]
        data_tmp_3 = data2.loc[data2.Air_group.isin(idx_3_values), ['Air_group'] + var_names]
    
        # Does not work because idx_x_values are not sorted!
        #data_tmp_1 = data2.loc[idx_1_values, ['Air_group'] + var_names]
        #data_tmp_2 = data2.loc[idx_2_values, ['Air_group'] + var_names]
        #data_tmp_3 = data2.loc[idx_3_values, ['Air_group'] + var_names]
    
        # Merges
        data_tmp_1.columns = [n + '_Entire home/apt' for n in ['Air_group'] + var_names]
        data_tmp_2.columns = [n + '_Private room' for n in ['Air_group'] + var_names]
        data_tmp_3.columns = [n + '_Shared room' for n in ['Air_group'] + var_names]
    
        tmp = df_key_value.merge(data_tmp_1, left_on='Air_group_data2_1',
                            right_on='Air_group_Entire home/apt', how='left')
        tmp.drop(columns='Air_group_Entire home/apt', inplace=True)
        tmp = tmp.merge(data_tmp_2, left_on='Air_group_data2_2',
                   right_on='Air_group_Private room', how='left')
        tmp.drop(columns='Air_group_Private room', inplace=True)
        tmp = tmp.merge(data_tmp_3, left_on='Air_group_data2_3',
                   right_on='Air_group_Shared room', how='left')
        tmp.drop(columns='Air_group_Shared room', inplace=True)
    
        #print(tmp.columns)
    
        if distance:
        
            dist_1 = []
            dist_2 = []
            dist_3 = []
        
            for g in data1.Air_group:
            
                if idx_1_dict[str(g)] is not None:
                    dist_1.append(_haversine_np(data1.loc[g, 'Air_longitude'], 
                                                    data1.loc[g, 'Air_latitude'],
                                                    data2.loc[idx_1_dict[str(g)], 'Air_longitude'], 
                                                    data2.loc[idx_1_dict[str(g)], 'Air_latitude']))
                else:
                    dist_1.append(None)
            
                if idx_2_dict[str(g)] is not None:
                    dist_2.append(_haversine_np(data1.loc[g, 'Air_longitude'], 
                                                    data1.loc[g, 'Air_latitude'],
                                                    data2.loc[idx_2_dict[str(g)], 'Air_longitude'], 
                                                    data2.loc[idx_2_dict[str(g)], 'Air_latitude']))
                else:
                    dist_2.append(None)
                
                if idx_3_dict[str(g)] is not None:
                    dist_3.append(_haversine_np(data1.loc[g, 'Air_longitude'], 
                                                data1.loc[g, 'Air_latitude'],
                                                data2.loc[idx_3_dict[str(g)], 'Air_longitude'], 
                                                data2.loc[idx_3_dict[str(g)], 'Air_latitude']))
                else:
                    dist_3.append(None)
        
            d_1 = {'dist': dist_1, 'Air_group': data1.Air_group}
            d_2 = {'dist': dist_2, 'Air_group': data1.Air_group}
            d_3 = {'dist': dist_3, 'Air_group': data1.Air_group}
        
            dist_1 = pd.DataFrame(data=d_1)
            dist_2 = pd.DataFrame(data=d_2)
            dist_3 = pd.DataFrame(data=d_3)
        
            data_tmp = tmp.merge(dist_1, left_on='Air_group_data1', right_on='Air_group',
                                 how='left')
            data_tmp.rename(columns={'dist': 'dist_Entire home/apt'}, inplace=True)
            #print(data_tmp.head(1))
            
            data_tmp = data_tmp.merge(dist_2, left_on='Air_group_data1', right_on='Air_group',
                                 how='left')
            data_tmp.rename(columns={'dist': 'dist_Private Room'}, inplace=True)
            #print(data_tmp.head(1))
            
            data_tmp = data_tmp.merge(dist_3, left_on='Air_group_data1', right_on='Air_group',
                                 how='left')
            data_tmp.rename(columns={'dist': 'dist_Shared Room'}, inplace=True)
            #print(data_tmp.head(1))
    
        # sort by Air_group
        data_tmp.sort_values(by='Air_group', inplace=True)
    
        # cleaning
        data_tmp.drop(columns=['Air_group', 'Air_group_x', 'Air_group_y', 
                               'Air_group_data2_1', 'Air_group_data2_2',
                              'Air_group_data2_3'], inplace=True)
        data_tmp.rename(columns={'Air_group_data1': 'Air_group'}, inplace=True)
    
    # Property type
    # ----------------------
    if classifier == "Air_property_type_2":
        
        # NEW: dictionaries instead of lists
        idx_1_dict = dict()
        idx_2_dict = dict()
        idx_3_dict = dict()
        idx_4_dict = dict()
    
        idx_1_values = []
        idx_2_values = []
        idx_3_values = []
        idx_4_values = []
    
        idx_1_keys = []
        idx_2_keys = []
        idx_3_keys = []
        idx_4_keys = []
    
        #'Other', 'House_Cottage_Villa', 
        #'Apartment_Condominium', 'Townhouse'
    
        for i in obs.keys():
            idx_1_dict[i] = obs[i]['Other']
            idx_2_dict[i] = obs[i]['House_Cottage_Villa']
            idx_3_dict[i] = obs[i]['Apartment_Condominium']
            idx_4_dict[i] = obs[i]['Townhouse']
        
            idx_1_values.append(obs[i]['Other'])
            idx_2_values.append(obs[i]['House_Cottage_Villa'])
            idx_3_values.append(obs[i]['Apartment_Condominium'])
            idx_4_values.append(obs[i]['Townhouse'])
        
            idx_1_keys.append(int(i))
            idx_2_keys.append(int(i))
            idx_3_keys.append(int(i))
            idx_4_keys.append(int(i))
    
        df_key_value_1 = pd.DataFrame({'Air_group_data1': idx_1_keys,
                                      'Air_group_data2_1': idx_1_values})
        df_key_value_2 = pd.DataFrame({'Air_group_data2_2': idx_2_values})
        df_key_value_3 = pd.DataFrame({'Air_group_data2_3': idx_3_values})
        df_key_value_4 = pd.DataFrame({'Air_group_data2_4': idx_4_values})
    
        df_key_value = pd.concat([df_key_value_1, df_key_value_2, 
                                  df_key_value_3, df_key_value_4], axis=1)
    
        data_tmp_1 = data2.loc[data2.Air_group.isin(idx_1_values), ['Air_group'] + var_names]
        data_tmp_2 = data2.loc[data2.Air_group.isin(idx_2_values), ['Air_group'] + var_names]
        data_tmp_3 = data2.loc[data2.Air_group.isin(idx_3_values), ['Air_group'] + var_names]
        data_tmp_4 = data2.loc[data2.Air_group.isin(idx_4_values), ['Air_group'] + var_names]
    
        # Does not work because idx_x_values are not sorted!
        #data_tmp_1 = data2.loc[idx_1_values, ['Air_group'] + var_names]
        #data_tmp_2 = data2.loc[idx_2_values, ['Air_group'] + var_names]
        #data_tmp_3 = data2.loc[idx_3_values, ['Air_group'] + var_names]
    
        # Merges
        data_tmp_1.columns = [n + '_Other' for n in ['Air_group'] + var_names]
        data_tmp_2.columns = [n + '_House_Cottage_Villa' for n in ['Air_group'] + var_names]
        data_tmp_3.columns = [n + '_Apartment_Condominium' for n in ['Air_group'] + var_names]
        data_tmp_4.columns = [n + '_Townhouse' for n in ['Air_group'] + var_names]
    
        tmp = df_key_value.merge(data_tmp_1, left_on='Air_group_data2_1',
                            right_on='Air_group_Other', how='left')
        tmp.drop(columns='Air_group_Other', inplace=True)
        tmp = tmp.merge(data_tmp_2, left_on='Air_group_data2_2',
                   right_on='Air_group_House_Cottage_Villa', how='left')
        tmp.drop(columns='Air_group_House_Cottage_Villa', inplace=True)
        tmp = tmp.merge(data_tmp_3, left_on='Air_group_data2_3',
                   right_on='Air_group_Apartment_Condominium', how='left')
        tmp.drop(columns='Air_group_Apartment_Condominium', inplace=True)
        tmp = tmp.merge(data_tmp_4, left_on='Air_group_data2_4',
                        right_on='Air_group_Townhouse', how='left')
        tmp.drop(columns='Air_group_Townhouse', inplace=True)
        
    
        #print(tmp.columns)
    
        if distance:
        
            dist_1 = []
            dist_2 = []
            dist_3 = []
            dist_4 = []
        
            for g in data1.Air_group:
            
                if idx_1_dict[str(g)] is not None:
                    dist_1.append(_haversine_np(data1.loc[g, 'Air_longitude'], 
                                                    data1.loc[g, 'Air_latitude'],
                                                    data2.loc[idx_1_dict[str(g)], 'Air_longitude'], 
                                                    data2.loc[idx_1_dict[str(g)], 'Air_latitude']))
                else:
                    dist_1.append(None)
            
                if idx_2_dict[str(g)] is not None:
                    dist_2.append(_haversine_np(data1.loc[g, 'Air_longitude'], 
                                                    data1.loc[g, 'Air_latitude'],
                                                    data2.loc[idx_2_dict[str(g)], 'Air_longitude'], 
                                                    data2.loc[idx_2_dict[str(g)], 'Air_latitude']))
                else:
                    dist_2.append(None)
                
                if idx_3_dict[str(g)] is not None:
                    dist_3.append(_haversine_np(data1.loc[g, 'Air_longitude'], 
                                                    data1.loc[g, 'Air_latitude'],
                                                    data2.loc[idx_3_dict[str(g)], 'Air_longitude'], 
                                                    data2.loc[idx_3_dict[str(g)], 'Air_latitude']))
                else:
                    dist_3.append(None)
                    
                if idx_4_dict[str(g)] is not None:
                    dist_4.append(_haversine_np(data1.loc[g, 'Air_longitude'],
                                                    data1.loc[g, 'Air_latitude'],
                                                    data2.loc[idx_4_dict[str(g)], 'Air_longitude'],
                                                    data2.loc[idx_4_dict[str(g)], 'Air_latitude']))
                else:
                    dist_4.append(None)
        
            d_1 = {'dist': dist_1, 'Air_group': data1.Air_group}
            d_2 = {'dist': dist_2, 'Air_group': data1.Air_group}
            d_3 = {'dist': dist_3, 'Air_group': data1.Air_group}
            d_4 = {'dist': dist_4, 'Air_group': data1.Air_group}
        
            dist_1 = pd.DataFrame(data=d_1)
            dist_2 = pd.DataFrame(data=d_2)
            dist_3 = pd.DataFrame(data=d_3)
            dist_4 = pd.DataFrame(data=d_4)
        
            #print(dist_4.head(5))
        
            data_tmp = tmp.merge(dist_1, left_on='Air_group_data1', right_on='Air_group',
                                 how='left')
            data_tmp.rename(columns={'dist': 'dist_Other'}, inplace=True)
            #print(data_tmp.head(1))
            
            data_tmp = data_tmp.merge(dist_2, left_on='Air_group_data1', right_on='Air_group',
                                 how='left')
            data_tmp.rename(columns={'dist': 'dist_House_Cottage_Villa'}, inplace=True)
            #print(data_tmp.head(1))
            
            data_tmp = data_tmp.merge(dist_3, left_on='Air_group_data1', right_on='Air_group',
                                 how='left')
            data_tmp.rename(columns={'dist': 'dist_Apartment_Condominium'}, inplace=True)
            #print(data_tmp.head(1))
            
            data_tmp = data_tmp.merge(dist_4, left_on='Air_group_data1', right_on='Air_group',
                                     how='left')
            data_tmp.rename(columns={'dist': 'dist_Townhouse'}, inplace=True)
            #print(data_tmp.head(1))
            
    
        # sort by Air_group
        data_tmp.sort_values(by='Air_group_data1', inplace=True)
    
        # cleaning
        try:
            data_tmp.drop(columns=['Air_group', 'Air_group_x', 'Air_group_y', 
                                   'Air_group_data2_1', 'Air_group_data2_2',
                                   'Air_group_data2_3', 'Air_group_data2_4'], inplace=True)
        except:
            data_tmp.drop(columns=['Air_group_x', 'Air_group_y', 
                                   'Air_group_data2_1', 'Air_group_data2_2',
                                   'Air_group_data2_3', 'Air_group_data2_4'], inplace=True)
            
        data_tmp.rename(columns={'Air_group_data1': 'Air_group'}, inplace=True)
    
    return data_tmp 
