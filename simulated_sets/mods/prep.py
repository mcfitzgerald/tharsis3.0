#preps raw data csv's for analysis using numpy arrays

#Takes CSV data file in graphpad table format --- first column contins all x-values
#and subsequent columns correspond to different y-values for different datasets thus 
#fields will be missing in a diagonal pattern when there are more than one dataset.
#Returns numpy arrays for further processing"


#dependencies
import numpy as np
import pandas as pd
from collections import OrderedDict as Ord

def makedf(dat):
    """Takes csv file location as string, returns pandas dataframe"""
    return pd.read_csv(dat)
    
def colist(df):
    """Takes dataframe returns columns headers excluding first column"""
    return df.columns.tolist()[1:]
    
def framelist(df,cols):
    """Splits dataframe containing multiple data sets into list of dataframes for 
    each dataset"""
    xvals = df.columns[0]
    return [df[[xvals, i]].dropna().sort_values(by=xvals) for i in cols]
    
def rtotinds(df):
    """Takes datafram and returns list of rtot idxs"""
    return df.iloc[:,0].tolist()
    
def checkidx(satdf,rtotdf):
    satidx = colist(satdf)
    rtotidx = rtotinds(rtotdf)
    
    #print(satidx, rtotidx)
    return satidx == rtotidx
    
def rtotlist(df):
    """Takes datafram and returns list of rtot values"""
    return df.iloc[:,1].values.tolist()


def datprep(satdat,rtotdat):
    """Takes csv data files (ligand binding data in first position, 'satdat',
    and receptor concentration data, 'rtotdat' in second position) and returns 
    numpy arrays of values for fitting functions.
    
    Need to work in description of data files, e.g., satdat is 
    ligand binding data in Graphpad xy-table style (all xvals in
    first column and yvals corresponding to diferent datasets in subsequent columns,
    and rtotdat is two columns where the first column should match the data set
    column header used in satdat, and the second column is the total receptor 
    concentration in the same units used in satdat, e.g., mmol.)
    """
    sats = makedf(satdat)
    rtots = makedf(rtotdat)
    
    if checkidx(sats,rtots) is True: 
        print('hootie hoo! the indices match!')
        
        data_labels = colist(sats) #note to self: i've computed colist twice now, 'chexkidx' func is costly
        
        rtotal_values = np.array(rtotlist(rtots))
        
        list_of_frames = framelist(sats,data_labels)
        
        lig_vals = np.array([i.iloc[:,0].values for i in list_of_frames])
        
        sat_vals = np.array([i.iloc[:,1].values for i in list_of_frames])
        
        return data_labels, rtotal_values, lig_vals, sat_vals
        
        
    else:
        print('oh poop (-_-) the indices don\'t match')
         
    

