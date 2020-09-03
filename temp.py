# -*- coding: utf-8 -*-
"""
@author: ethan
"""
import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen

#data_path = 'C:/Users/ethan/Documents/MEE/Research/Rcode/snap_data_clean.csv'
data_path = 'C:/Users/ethan/Documents/MEE/Research/snap_data3.csv'

raw_data = pd.read_csv(data_path, encoding = "ISO-8859-1")

data = raw_data.copy(deep=True)
col_names = (list(data.columns))[2:]

user_list = data['ID'].drop_duplicates().tolist()
user_dfs = [data.loc[data['ID'] == us_id] for us_id in user_list ]


def scale_and_clean(arr, verbose=True):
    stdv = np.sqrt((np.var(arr)))
    new_arr = np.array(arr)
    new_arr = new_arr - np.mean(new_arr)
    if(stdv != 0):
        new_arr = new_arr/stdv
    elif(verbose):
        print("WARNING: variance = 0")

    return new_arr


#%%

for df in user_dfs:
    for col in col_names:
        df[col] = scale_and_clean(df[col])
        
#%%

def get_skewness(samp_arr, verbose=False):
    stdv = np.sqrt(np.var(samp_arr))
    skew = 0
    if(stdv != 0):
        skew = (np.median(samp_arr) - np.mean(samp_arr))/stdv
    elif(verbose):
        print("WARNING: variance = 0")
    return skew

def skewness_inv(samp_arr, epsilon=1e-12):
    return np.abs(1/(get_skewness(samp_arr) + epsilon))

def repeat_arr(arr, N):
    new_arr = []
    for i in range(N):
        new_arr.append(arr)
    return new_arr

def select_max(N, Sn, func=get_skewness):
    max_arr = np.random.uniform(-1, 1, Sn)
    max_val = func(max_arr)
    for i in range(N):
        sarr = np.random.uniform(-1, 1, Sn)
        new_val = func(sarr)
        if(new_val > max_val):
            max_val = new_val
            max_arr = sarr
    return max_arr

def shift_left_app(arr, new_val):
    new_arr = arr[1:]
    new_arr.append(new_val)
    return new_arr



def L1_sum(arr):
    return(np.sum(np.abs(arr)))
def L2_sum(arr):
    return np.sqrt(np.sum(np.multiply(arr, np.conj(arr))))

def test_func(arr):
    val = L2_sum(np.diff(np.diff(arr)))
    return val

def select_max_grad(N, Sn, delt = 0.01, func=get_skewness, max_sum=None, sum_func=L1_sum):
    eps = 1e-12
    arr = np.random.uniform(-1, 1, Sn)
    delt = 0.01
    max_val = func(arr)
    prev_dir = np.zeros(Sn)
    del_inc = 0
    inc_arr = [0,0,0,0,0]
    for i in range(N):
        sarr = np.random.uniform(-1, 1, Sn)
        cur_dir = delt*sarr
        #jit_dir = ((1)*(((del_inc + eps)/(np.mean(inc_arr) + eps))))*prev_dir
        
        #new_arr = arr + cur_dir + jit_dir
        new_arr = arr + cur_dir
        new_val = func(new_arr)
        
        rand_guess = np.random.uniform(-1, 1, Sn)
        rand_val = func(rand_guess)
        
        if(rand_val > new_val):
            new_val = rand_val
            new_arr = rand_guess
            
        if(new_val > max_val):
            if(max_sum is not None):
                if(sum_func(new_arr) > max_sum):
                    continue
            del_inc = new_val - max_val
            prev_dir = cur_dir
            max_val = new_val
            arr = new_arr
            inc_arr = shift_left_app(inc_arr, del_inc)
            
    return arr
    

def cal_mse(pred_arr, actual_arr):
    par = np.array(pred_arr)
    aar = np.array(actual_arr)
    N = len(pred_arr)
    mse = np.sum((1/N)*(np.power(par-aar, 2)))
    return mse
#%%
maxlag=3
test = 'ssr_chi2test'
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            #print([r,c])
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

def cointegration_test(df, alpha=0.05): 
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)



col_names = ['morning_stress', 'sleep_duration', 'academic_duration',
       'awakening_duration', 'nap_duration', 'time_in_bed',
       'call_total_duration', 'screen_total_duration']
col_names = [col_names[i] for i in [0,1,2,5,6]]
test_df = (user_dfs[2])[col_names]

#gc_res = grangers_causation_matrix(test_df, variables = test_df.columns)
#gc_avg = pd.DataFrame(sum([np.array(grangers_causation_matrix(df[col_names], variables = df[col_names].columns)) for df in user_dfs])/len(user_dfs), columns=col_names, index=col_names)

ci_res = cointegration_test(test_df)

#%%
def calmat(H):
    HT = np.conj(np.transpose(H))
    C = np.matmul(H, HT)
    Ci = np.linalg.pinv(C)
    T = np.matmul(Ci, H)
    return np.matmul(HT, T)


